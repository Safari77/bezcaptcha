#!/usr/bin/env python3
import base64
import io
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import binary_dilation, label

# Matches BASE32_ALPHABET in src/captcha.rs (exact case per character)
BASE32_ALPHABET = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "B",
    "C",
    "d",
    "e",
    "F",
    "G",
    "h",
    "J",
    "K",
    "m",
    "n",
    "P",
    "q",
    "r",
    "S",
    "t",
    "u",
    "V",
    "W",
    "X",
    "y",
    "Z",
]
LEFT_PORT_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8"]

WIDTH = 460
HEIGHT = 270


class BezierCaptchaSolver:
    def __init__(self, font_path: Optional[str] = None):
        # Each character maps to a list of pre-warped, centered boolean templates
        self.badge_templates: Dict[str, np.ndarray] = {}
        self.port_templates: Dict[str, np.ndarray] = {}
        if font_path and os.path.exists(font_path):
            self._load_font_templates(font_path)

    def _normalize_glyph(
        self, mask: np.ndarray, canvas_size: int = 24
    ) -> Optional[np.ndarray]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()
        crop = mask[min_y : max_y + 1, min_x : max_x + 1]
        h, w = crop.shape
        if h > canvas_size or w > canvas_size:
            crop = crop[: min(h, canvas_size), : min(w, canvas_size)]
            h, w = crop.shape
        out = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        y0 = (canvas_size - h) // 2
        x0 = (canvas_size - w) // 2
        out[y0 : y0 + h, x0 : x0 + w] = crop.astype(np.float32)
        return out

    def _load_font_templates(self, font_path: str):
        from PIL import ImageDraw, ImageFont

        # ab_glyph PxScale(20.0) and PxScale(19.5) correspond to ~16-18pt in FreeType/PIL
        all_chars = list(set(BASE32_ALPHABET + LEFT_PORT_LABELS))

        for target_dict, sizes in [
            (self.badge_templates, [15, 16, 17, 18]),
            (self.port_templates, [16, 17, 18, 19]),
        ]:
            for ch in all_chars:
                variants = []
                for sz in sizes:
                    font = ImageFont.truetype(font_path, sz)
                    img = Image.new("L", (40, 40), 0)
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 8), ch, font=font, fill=255)
                    arr = np.array(img)
                    ys, xs = np.where(arr > 64)
                    if len(xs) == 0:
                        continue
                    cy, cx = 0.5 * (ys.min() + ys.max()), 0.5 * (xs.min() + xs.max())

                    # Generate sheared and rotated variants matching draw_distorted_glyph_mut
                    for rot_deg in [-10.0, -5.0, 0.0, 5.0, 10.0]:
                        theta = np.deg2rad(rot_deg)
                        cos_t, sin_t = np.cos(theta), np.sin(theta)
                        for shx_deg in [-15.0, -8.0, 0.0, 8.0, 15.0]:
                            shx = np.tan(np.deg2rad(shx_deg))
                            m00 = cos_t
                            m01 = cos_t * shx - sin_t
                            m10 = sin_t
                            m11 = sin_t * shx + cos_t
                            det = m00 * m11 - m01 * m10
                            if abs(det) < 1e-4:
                                continue
                            inv00, inv01 = m11 / det, -m01 / det
                            inv10, inv11 = -m10 / det, m00 / det
                            c_off = cx - inv00 * cx - inv01 * cy
                            f_off = cy - inv10 * cx - inv11 * cy
                            warped = img.transform(
                                (40, 40),
                                Image.Transform.AFFINE,
                                (inv00, inv01, c_off, inv10, inv11, f_off),
                                resample=Image.Resampling.BILINEAR,
                            )
                            w_arr = np.array(warped) > 105
                            normed = self._normalize_glyph(w_arr)
                            if normed is not None:
                                variants.append(normed)
                if variants:
                    target_dict[ch] = np.stack(variants, axis=0)

    def parse_challenge_prompt(self, prompt: str) -> Tuple[List[str], int]:
        """
        Parses wires to trace and badges expected per wire:
        e.g., 'Trace wires 3 and 4: enter each wire's 3 characters from left to right ... (8 chars, ...)'
        """
        match_wires = re.search(r"Trace wires ([\d\s,and]+):", prompt)
        if match_wires:
            raw = match_wires.group(1).replace("and", ",").split(",")
            queried = [w.strip() for w in raw if w.strip()]
        else:
            queried = ["1", "2"]

        match_chars = re.search(
            r"enter each wire's (\d+) (?:characters|digits)", prompt
        )
        if match_chars:
            badges_per_wire = int(match_chars.group(1))
        elif "waypoint character" in prompt or "waypoint digit" in prompt:
            badges_per_wire = 1
        else:
            match_total = re.search(r"\((\d+)\s*chars", prompt)
            if match_total and len(queried) > 0:
                badges_per_wire = max(1, int(match_total.group(1)) // len(queried) - 1)
            else:
                badges_per_wire = 1

        return queried, badges_per_wire

    def extract_binary_masks(
        self, img_pil: Image.Image
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.array(img_pil.convert("RGB"), dtype=np.float32)
        # Background is RGB(248, 249, 250); strokes/text are RGB(35, 35, 40)
        bg = np.array([248.0, 249.0, 250.0])
        dist_from_bg = np.linalg.norm(arr - bg, axis=2)

        stroke_mask = dist_from_bg > 50.0
        core_stroke = dist_from_bg > 110.0
        return stroke_mask, core_stroke, dist_from_bg

    def _score_crop_candidates(
        self,
        crop: np.ndarray,
        templates: Dict[str, np.ndarray],
        candidates: List[str],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        norm_crop = self._normalize_glyph(crop)
        if norm_crop is None:
            return {ch: 0.0 for ch in candidates}

        # Allow small 1px translations to absorb rounding differences
        shifted_crops = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted = np.roll(norm_crop, (dy, dx), axis=(0, 1))
                shifted_crops.append(shifted)
        crops_stack = np.stack(shifted_crops, axis=0)  # (9, 24, 24)
        ch_h, ch_w = crop.shape
        aspect = ch_w / max(ch_h, 1)

        for ch in candidates:
            if ch not in templates:
                scores[ch] = 0.0
                continue
            tpls = templates[ch]  # (N, 24, 24)
            # Compute IoU across all templates and shifts using matrix multiplication
            c_flat = crops_stack.reshape(9, -1)  # (9, 576)
            t_flat = tpls.reshape(tpls.shape[0], -1)  # (N, 576)
            inter = np.dot(c_flat, t_flat.T)  # (9, N)
            union = (
                c_flat.sum(axis=1, keepdims=True)
                + t_flat.sum(axis=1, keepdims=True).T
                - inter
            )
            iou = float(np.max(inter / np.maximum(union, 1.0)))

            # Aspect ratio sanity check for narrow vs wide glyphs
            if ch == "1" and aspect > 0.72:
                iou *= 0.85
            scores[ch] = iou

        return scores

    def _match_crop(
        self, crop: np.ndarray, templates: Dict[str, np.ndarray], candidates: List[str]
    ) -> Optional[str]:
        scores = self._score_crop_candidates(crop, templates, candidates)
        if not scores:
            return None
        best_char = max(scores, key=scores.get)
        return best_char if scores[best_char] > 0.18 else None

    def _find_inverted_circles(self, dist_from_bg: np.ndarray) -> List[Tuple[int, int]]:
        """
        Finds centers (cx, cy) of radius-10 dark filled circles used for inverted-contrast
        badges and ports. A radius-10 circle has area ~314 px and a solid dark ring at r in [7, 9.5].
        """
        yy, xx = np.ogrid[-11:12, -11:12]
        r2 = xx * xx + yy * yy
        ring_kernel = ((r2 >= 45) & (r2 <= 96)).astype(np.float32)
        ring_kernel /= ring_kernel.sum()

        outer_kernel = ((r2 >= 115) & (r2 <= 145)).astype(np.float32)
        outer_kernel /= outer_kernel.sum()

        from scipy.ndimage import convolve, maximum_filter

        dark = (dist_from_bg > 110.0).astype(np.float32)
        ring_score = convolve(dark, ring_kernel, mode="constant", cval=0.0)
        outer_score = convolve(dark, outer_kernel, mode="constant", cval=0.0)

        # Ring must be predominantly dark while outside the radius-10 circle is mostly clear
        score = ring_score - 0.65 * outer_score
        local_max = (
            (score == maximum_filter(score, size=15))
            & (ring_score > 0.72)
            & (outer_score < 0.45)
        )

        ys, xs = np.where(local_max)
        centers = []
        for y, x in zip(ys, xs):
            # Also verify there are light glyph pixels inside the core (r <= 6)
            patch = dist_from_bg[
                max(0, y - 6) : min(HEIGHT, y + 7), max(0, x - 6) : min(WIDTH, x + 7)
            ]
            if np.any(patch < 140.0):
                centers.append((int(x), int(y)))
        return centers

    def _extract_glyph_at(
        self, cx: int, cy: int, dist_from_bg: np.ndarray, inverted: bool
    ) -> Optional[np.ndarray]:
        r = 9 if inverted else 10
        y0, y1 = max(0, cy - r), min(HEIGHT, cy + r + 1)
        x0, x1 = max(0, cx - r), min(WIDTH, cx + r + 1)
        patch = dist_from_bg[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0 - cy : y1 - cy, x0 - cx : x1 - cx]
        disk = (xx * xx + yy * yy) <= (72 if inverted else 95)

        if inverted:
            # Light glyph on dark circle: pixels inside disk that are brighter than dark stroke
            glyph = (patch < 145.0) & disk
        else:
            # Dark glyph on light background
            glyph = (patch > 85.0) & disk

        if not np.any(glyph):
            return None

        # Keep connected components near the center
        labeled, n_feat = label(glyph)
        if n_feat > 1:
            keep = np.zeros_like(glyph)
            for idx in range(1, n_feat + 1):
                comp = labeled == idx
                cys, cxs = np.where(comp)
                if (
                    len(cxs) >= 2
                    and np.hypot(cxs.mean() - (cx - x0), cys.mean() - (cy - y0)) <= 6.5
                ):
                    keep |= comp
            if np.any(keep):
                glyph = keep

        ys, xs = np.where(glyph)
        if len(xs) < 4:
            return None
        return glyph[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    def locate_ports(
        self, stroke_mask: np.ndarray, dist_from_bg: np.ndarray
    ) -> Tuple[
        Dict[str, Tuple[float, float]],
        Dict[str, Tuple[float, float]],
        List[Tuple[int, int]],
    ]:
        inverted_centers = self._find_inverted_circles(dist_from_bg)

        # Mask out inverted circles to find normal-contrast port glyphs cleanly
        clean_stroke = stroke_mask.copy()
        for icx, icy in inverted_centers:
            y0, y1 = max(0, icy - 11), min(HEIGHT, icy + 12)
            x0, x1 = max(0, icx - 11), min(WIDTH, icx + 12)
            yy, xx = np.ogrid[y0 - icy : y1 - icy, x0 - icx : x1 - icx]
            clean_stroke[y0:y1, x0:x1][xx * xx + yy * yy <= 121] = False

        left_items = []
        right_items = []

        # 1. Inverted-contrast ports
        for icx, icy in inverted_centers:
            if icx < 32:
                crop = self._extract_glyph_at(icx, icy, dist_from_bg, inverted=True)
                if crop is not None:
                    left_items.append((float(icy), float(icx), crop))
            elif icx > WIDTH - 32:
                crop = self._extract_glyph_at(icx, icy, dist_from_bg, inverted=True)
                if crop is not None:
                    right_items.append((float(icy), float(icx), crop))

        # 2. Normal-contrast ports
        labeled, num_features = label(clean_stroke)
        for feat_idx in range(1, num_features + 1):
            ys, xs = np.where(labeled == feat_idx)
            if len(xs) < 10 or len(xs) > 450:
                continue
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            min_x, max_x = np.min(xs), np.max(xs)
            min_y, max_y = np.min(ys), np.max(ys)
            h = max_y - min_y + 1
            w = max_x - min_x + 1

            if cx < 30.0 and 8 <= h <= 26 and w <= 22:
                if any(abs(cy - ey) < 14.0 for ey, _, _ in left_items):
                    continue
                crop = clean_stroke[min_y : max_y + 1, min_x : max_x + 1]
                left_items.append((cy, cx, crop))
            elif cx > 430.0 and 8 <= h <= 26 and w <= 22:
                if any(abs(cy - ey) < 14.0 for ey, _, _ in right_items):
                    continue
                crop = clean_stroke[min_y : max_y + 1, min_x : max_x + 1]
                right_items.append((cy, cx, crop))

        left_items.sort(key=lambda item: item[0])
        right_items.sort(key=lambda item: item[0])

        # Left ports: labels are 1..N (where N = len(left_items)), vertically shuffled!
        # Use greedy maximum-weight bipartite assignment over 1..N
        n_left = len(left_items)
        active_left_labels = LEFT_PORT_LABELS[: max(n_left, 5)]
        left_candidates = []
        for idx, (cy, cx, crop) in enumerate(left_items):
            scores = self._score_crop_candidates(
                crop, self.port_templates, active_left_labels
            )
            for lbl, sc in scores.items():
                left_candidates.append((sc, idx, lbl))

        left_candidates.sort(reverse=True, key=lambda x: x[0])
        assigned_left_idx = set()
        left_ports: Dict[str, Tuple[float, float]] = {}
        for sc, idx, lbl in left_candidates:
            if idx not in assigned_left_idx and lbl not in left_ports:
                cy, cx, _ = left_items[idx]
                # Generator draws glyph at cx = lx + 5 and attaches wire at p0 = lx + 18 -> cx + 13
                left_ports[lbl] = (cx + 13.0, cy)
                assigned_left_idx.add(idx)

        for idx, (cy, cx, _) in enumerate(left_items):
            if idx not in assigned_left_idx:
                for lbl in active_left_labels:
                    if lbl not in left_ports:
                        left_ports[lbl] = (cx + 13.0, cy)
                        break

        # Right ports: randomly chosen unique characters from BASE32_ALPHABET in generate_path_captcha_impl
        right_candidates = []
        for idx, (cy, cx, crop) in enumerate(right_items):
            scores = self._score_crop_candidates(
                crop, self.port_templates, BASE32_ALPHABET
            )
            for lbl, sc in scores.items():
                right_candidates.append((sc, idx, lbl))

        right_candidates.sort(reverse=True, key=lambda x: x[0])
        assigned_right_idx = set()
        right_ports: Dict[str, Tuple[float, float]] = {}
        for sc, idx, lbl in right_candidates:
            if idx not in assigned_right_idx and lbl not in right_ports:
                cy, cx, _ = right_items[idx]
                # Generator draws glyph at cx = rx + 5 and attaches wire at p3 = rx - 8 -> cx - 13
                right_ports[lbl] = (cx - 13.0, cy)
                assigned_right_idx.add(idx)

        for idx, (cy, cx, _) in enumerate(right_items):
            if idx not in assigned_right_idx:
                for lbl in BASE32_ALPHABET:
                    if lbl not in right_ports:
                        right_ports[lbl] = (cx - 13.0, cy)
                        break

        return left_ports, right_ports, inverted_centers

    def find_all_badges(
        self,
        stroke_mask: np.ndarray,
        dist_from_bg: np.ndarray,
        inverted_centers: List[Tuple[int, int]],
        excluded_chars: Optional[set] = None,
    ) -> List[Dict]:
        """
        Locates both normal-contrast and inverted-contrast waypoint badges in the central area.
        """
        badges: List[Dict] = []
        candidate_alphabet = [
            c for c in BASE32_ALPHABET if not excluded_chars or c not in excluded_chars
        ]
        if not candidate_alphabet:
            candidate_alphabet = BASE32_ALPHABET

        # 1. Inverted-contrast badges (radius 10 dark circle with light glyph at (badge.x, badge.y))
        clean_stroke = stroke_mask.copy()
        for icx, icy in inverted_centers:
            y0, y1 = max(0, icy - 11), min(HEIGHT, icy + 12)
            x0, x1 = max(0, icx - 11), min(WIDTH, icx + 12)
            yy, xx = np.ogrid[y0 - icy : y1 - icy, x0 - icx : x1 - icx]
            clean_stroke[y0:y1, x0:x1][xx * xx + yy * yy <= 125] = False

            if 36 <= icx <= WIDTH - 36:
                crop = self._extract_glyph_at(icx, icy, dist_from_bg, inverted=True)
                if crop is not None:
                    scores = self._score_crop_candidates(
                        crop, self.badge_templates, candidate_alphabet
                    )
                    best_ch = max(scores, key=scores.get)
                    badges.append(
                        {
                            "char": best_ch,
                            "scores": scores,
                            "x": float(icx),
                            "y": float(icy),
                            "inverted": True,
                        }
                    )

        # 2. Normal-contrast badges: dark glyph isolated inside a radius-10.5 light halo
        labeled, num_features = label(clean_stroke)
        for feat_idx in range(1, num_features + 1):
            ys, xs = np.where(labeled == feat_idx)
            if len(xs) < 6 or len(xs) > 260:
                continue

            min_x, max_x = int(np.min(xs)), int(np.max(xs))
            min_y, max_y = int(np.min(ys)), int(np.max(ys))
            h = max_y - min_y + 1
            w = max_x - min_x + 1
            cx = 0.5 * (min_x + max_x)
            cy = 0.5 * (min_y + max_y)

            if cx <= 38.0 or cx >= WIDTH - 38.0 or cy <= 14.0 or cy >= HEIGHT - 14.0:
                continue

            if 6 <= h <= 24 and 3 <= w <= 22:
                # Verify that the component is surrounded by a clear halo (radius ~9-10.5)
                # and isn't just a short background micro-stroke segment
                icx, icy = int(round(cx)), int(round(cy))
                y0, y1 = max(0, icy - 11), min(HEIGHT, icy + 12)
                x0, x1 = max(0, icx - 11), min(WIDTH, icx + 12)
                yy, xx = np.ogrid[y0 - icy : y1 - icy, x0 - icx : x1 - icx]
                r2 = xx * xx + yy * yy
                moat = (r2 >= 68) & (r2 <= 108)
                moat_pixels = clean_stroke[y0:y1, x0:x1][moat]
                if len(moat_pixels) > 0 and np.mean(moat_pixels) > 0.12:
                    continue

                if any(np.hypot(cx - b["x"], cy - b["y"]) < 16.0 for b in badges):
                    continue

                crop = clean_stroke[min_y : max_y + 1, min_x : max_x + 1]
                scores = self._score_crop_candidates(
                    crop, self.badge_templates, candidate_alphabet
                )
                best_ch = max(scores, key=scores.get)
                if scores[best_ch] > 0.20:
                    badges.append(
                        {
                            "char": best_ch,
                            "scores": scores,
                            "x": float(cx),
                            "y": float(cy),
                            "inverted": False,
                        }
                    )

        return badges

    def _build_wire_mask(
        self,
        core_stroke: np.ndarray,
        badges: List[Dict],
        left_ports: Dict[str, Tuple[float, float]],
        right_ports: Dict[str, Tuple[float, float]],
    ) -> np.ndarray:
        """
        Builds a clean wire-only boolean mask by erasing badge glyphs/circles and port glyphs.
        """
        wire_mask = core_stroke.copy()

        for b in badges:
            bx, by = int(round(b["x"])), int(round(b["y"]))
            y0, y1 = max(0, by - 11), min(HEIGHT, by + 12)
            x0, x1 = max(0, bx - 11), min(WIDTH, bx + 12)
            yy, xx = np.ogrid[y0 - by : y1 - by, x0 - bx : x1 - bx]
            wire_mask[y0:y1, x0:x1][xx * xx + yy * yy <= 112] = False

        for px, py in left_ports.values():
            lx, ly = int(round(px - 13.0)), int(round(py))
            y0, y1 = max(0, ly - 12), min(HEIGHT, ly + 13)
            x0, x1 = max(0, lx - 12), min(WIDTH, lx + 13)
            yy, xx = np.ogrid[y0 - ly : y1 - ly, x0 - lx : x1 - lx]
            wire_mask[y0:y1, x0:x1][xx * xx + yy * yy <= 125] = False

        for px, py in right_ports.values():
            rx, ry = int(round(px + 13.0)), int(round(py))
            y0, y1 = max(0, ry - 12), min(HEIGHT, ry + 13)
            x0, x1 = max(0, rx - 12), min(WIDTH, rx + 13)
            yy, xx = np.ogrid[y0 - ry : y1 - ry, x0 - rx : x1 - rx]
            wire_mask[y0:y1, x0:x1][xx * xx + yy * yy <= 125] = False

        return wire_mask

    def trace_wire(
        self,
        start_pt: Tuple[float, float],
        wire_mask: np.ndarray,
        all_badges: List[Dict],
        right_ports: Dict[str, Tuple[float, float]],
        expected_digits: int,
    ) -> Tuple[List[str], str]:
        """
        Traces a wire from start_pt across the canvas using a multi-hypothesis beam search
        that bridges chirp-modulated dash gaps (up to 15px), underpass cuts (13px + swerves),
        and badge halos (radius 10.5px with solid entry/exit locks at radius 10.5..16px).
        """
        dilated_wire = binary_dilation(wire_mask, iterations=1)

        # Estimate initial tangent near start_pt
        sx, sy = start_pt
        init_vx, init_vy = 1.0, 0.0
        best_init_count = -1
        for ang in np.linspace(-0.85, 0.85, 25):
            ca, sa = np.cos(ang), np.sin(ang)
            hits = 0
            for r in np.linspace(2.0, 18.0, 17):
                ix, iy = int(round(sx + r * ca)), int(round(sy + r * sa))
                if 0 <= ix < WIDTH and 0 <= iy < HEIGHT and dilated_wire[iy, ix]:
                    hits += 1
            if hits > best_init_count:
                best_init_count = hits
                init_vx, init_vy = ca, sa

        # Beam state: (score, x, y, vx, vy, badge_indices_tuple, history_list)
        beam = [(0.0, sx, sy, init_vx, init_vy, (), [(sx, sy)])]
        completed = []

        step_size = 4.0
        max_iters = 160

        for _ in range(max_iters):
            new_beam = []
            for score, x, y, vx, vy, b_tup, hist in beam:
                if x >= WIDTH - 34.0:
                    completed.append((score, x, y, vx, vy, b_tup, hist))
                    continue

                base_ang = np.arctan2(vy, vx)

                # Check if we are about to enter an unvisited badge halo (radius ~10.5px)
                jumped_badge = False
                for b_idx, b in enumerate(all_badges):
                    if b_idx in b_tup:
                        continue
                    dx_b = b["x"] - x
                    dy_b = b["y"] - y
                    dist_b = np.hypot(dx_b, dy_b)
                    if dist_b <= 15.0 and (dx_b * vx + dy_b * vy) > 0:
                        # Find exit lock on the far side of the badge halo (radius 11..16px)
                        in_ang = np.arctan2(dy_b, dx_b)
                        for da in np.linspace(-0.55, 0.55, 13):
                            out_ang = in_ang + da
                            if np.cos(out_ang) < -0.15:
                                continue
                            ca, sa = np.cos(out_ang), np.sin(out_ang)
                            lock_hits = 0
                            for r in [11.5, 13.0, 14.5, 16.0, 18.0]:
                                ix = int(round(b["x"] + r * ca))
                                iy = int(round(b["y"] + r * sa))
                                if (
                                    0 <= ix < WIDTH
                                    and 0 <= iy < HEIGHT
                                    and dilated_wire[iy, ix]
                                ):
                                    lock_hits += 1
                            if lock_hits >= 2:
                                nx = b["x"] + 13.5 * ca
                                ny = b["y"] + 13.5 * sa
                                n_score = score + 18.0 + lock_hits * 2.5 - abs(da) * 6.0
                                new_beam.append(
                                    (
                                        n_score,
                                        nx,
                                        ny,
                                        ca,
                                        sa,
                                        b_tup + (b_idx,),
                                        hist + [(b["x"], b["y"]), (nx, ny)],
                                    )
                                )
                                jumped_badge = True

                if jumped_badge:
                    continue

                # Standard step or gap jump
                candidates_for_state = []
                for da in np.linspace(-0.42, 0.42, 11):
                    ang = base_ang + da
                    ca, sa = np.cos(ang), np.sin(ang)
                    if ca < -0.15:  # Wires move generally left-to-right
                        continue

                    # Check immediate short step (1..5 px)
                    short_hits = 0
                    for r in [1.5, 3.0, 4.5]:
                        ix, iy = int(round(x + r * ca)), int(round(y + r * sa))
                        if (
                            0 <= ix < WIDTH
                            and 0 <= iy < HEIGHT
                            and dilated_wire[iy, ix]
                        ):
                            short_hits += 1

                    if short_hits >= 2:
                        nx, ny = x + step_size * ca, y + step_size * sa
                        nvx, nvy = 0.55 * vx + 0.45 * ca, 0.55 * vy + 0.45 * sa
                        norm = np.hypot(nvx, nvy)
                        nvx, nvy = nvx / norm, nvy / norm
                        step_sc = score + short_hits * 1.5 - abs(da) * 4.5
                        candidates_for_state.append(
                            (step_sc, nx, ny, nvx, nvy, b_tup, hist + [(nx, ny)])
                        )

                if candidates_for_state:
                    candidates_for_state.sort(reverse=True, key=lambda item: item[0])
                    new_beam.extend(candidates_for_state[:3])
                else:
                    # Gap bridging: dash gap (up to 15px) or underpass cut + dipole swerve (up to 22px)
                    gap_candidates = []
                    for jump_dist in [8.0, 11.5, 15.0, 19.0, 23.0]:
                        for da in np.linspace(-0.38, 0.38, 13):
                            ang = base_ang + da
                            ca, sa = np.cos(ang), np.sin(ang)
                            if ca < -0.1:
                                continue
                            jx, jy = x + jump_dist * ca, y + jump_dist * sa
                            if not (34 <= jx < WIDTH - 18 and 10 <= jy < HEIGHT - 10):
                                continue
                            ix, iy = int(round(jx)), int(round(jy))
                            if not dilated_wire[iy, ix]:
                                continue

                            # Verify tangent alignment of the landing segment
                            align_hits = 0
                            for r in [2.0, 4.0, 6.0]:
                                tx, ty = (
                                    int(round(jx + r * ca)),
                                    int(round(jy + r * sa)),
                                )
                                if (
                                    0 <= tx < WIDTH
                                    and 0 <= ty < HEIGHT
                                    and dilated_wire[ty, tx]
                                ):
                                    align_hits += 1

                            if align_hits >= 2:
                                nvx, nvy = 0.65 * vx + 0.35 * ca, 0.65 * vy + 0.35 * sa
                                norm = np.hypot(nvx, nvy)
                                nvx, nvy = nvx / norm, nvy / norm
                                gap_sc = (
                                    score
                                    + align_hits * 1.8
                                    - jump_dist * 0.25
                                    - abs(da) * 9.0
                                )
                                gap_candidates.append(
                                    (gap_sc, jx, jy, nvx, nvy, b_tup, hist + [(jx, jy)])
                                )

                    if gap_candidates:
                        gap_candidates.sort(reverse=True, key=lambda item: item[0])
                        new_beam.extend(gap_candidates[:3])

            if not new_beam:
                break

            # Deduplicate nearby states in (x, y, vx, vy, b_tup) space and keep top beam_width
            new_beam.sort(reverse=True, key=lambda item: item[0])
            pruned = []
            for cand in new_beam:
                sc, cx, cy, cvx, cvy, cb, chist = cand
                is_dup = False
                for _, px, py, pvx, pvy, pb, _ in pruned:
                    if (
                        cb == pb
                        and np.hypot(cx - px, cy - py) < 5.5
                        and (cvx * pvx + cvy * pvy) > 0.92
                    ):
                        is_dup = True
                        break
                if not is_dup:
                    pruned.append(cand)
                if len(pruned) >= 18:
                    break
            beam = pruned

        all_final = completed + beam
        if not all_final:
            all_final = [(0.0, sx, sy, 1.0, 0.0, (), [(sx, sy)])]

        # Score final trajectories by port proximity, badge count match, and path smoothness
        best_traj = None
        best_total_score = -1e9
        for sc, fx, fy, fvx, fvy, b_tup, hist in all_final:
            port_dist = min(
                (np.hypot(fx - px, fy - py) for px, py in right_ports.values()),
                default=50.0,
            )
            badge_bonus = 35.0 * min(len(b_tup), expected_digits)
            if len(b_tup) == expected_digits:
                badge_bonus += 25.0
            reach_bonus = 40.0 if fx >= WIDTH - 42.0 else (fx / WIDTH) * 15.0
            total_sc = sc + badge_bonus + reach_bonus - port_dist * 1.4
            if total_sc > best_total_score:
                best_total_score = total_sc
                best_traj = (fx, fy, b_tup, hist)

        fx, fy, b_tup, hist = best_traj
        path_arr = np.array(hist)

        # Collect badges along the winning trajectory
        collected_indices = list(b_tup)
        if len(collected_indices) < expected_digits:
            # Find closest badges to the traced path within each longitudinal band
            for b_idx, b in enumerate(all_badges):
                if b_idx in collected_indices:
                    continue
                dists = np.hypot(path_arr[:, 0] - b["x"], path_arr[:, 1] - b["y"])
                if np.min(dists) <= 18.0:
                    collected_indices.append(b_idx)

        # If still fewer than expected_digits, pick closest badge in the missing longitudinal band
        if len(collected_indices) < expected_digits:
            band_span = 0.68 / max(expected_digits, 1)
            for band in range(expected_digits):
                b_min_x = (0.16 + band * band_span) * WIDTH
                b_max_x = (0.16 + (band + 1) * band_span) * WIDTH
                if any(
                    b_min_x - 15 <= all_badges[i]["x"] <= b_max_x + 15
                    for i in collected_indices
                ):
                    continue
                best_b_idx = None
                best_d = 45.0
                for b_idx, b in enumerate(all_badges):
                    if b_idx in collected_indices:
                        continue
                    if b_min_x - 25 <= b["x"] <= b_max_x + 25:
                        d = float(
                            np.min(
                                np.hypot(
                                    path_arr[:, 0] - b["x"], path_arr[:, 1] - b["y"]
                                )
                            )
                        )
                        if d < best_d:
                            best_d = d
                            best_b_idx = b_idx
                if best_b_idx is not None:
                    collected_indices.append(best_b_idx)

        # If more than expected_digits were collected, keep the ones closest to the path
        if len(collected_indices) > expected_digits:
            collected_indices.sort(
                key=lambda idx: float(
                    np.min(
                        np.hypot(
                            path_arr[:, 0] - all_badges[idx]["x"],
                            path_arr[:, 1] - all_badges[idx]["y"],
                        )
                    )
                )
            )
            collected_indices = collected_indices[:expected_digits]

        # Sort badges strictly left-to-right by X coordinate
        collected_indices.sort(key=lambda idx: all_badges[idx]["x"])
        digits = [all_badges[idx]["char"] for idx in collected_indices]

        # Match destination port with closest right port attachment point
        best_port = BASE32_ALPHABET[0]
        min_dist = float("inf")
        for label_char, port_pt in right_ports.items():
            d = np.hypot(fx - port_pt[0], fy - port_pt[1])
            if d < min_dist:
                min_dist = d
                best_port = label_char

        return digits[:expected_digits], best_port

    def solve(self, image_data: bytes, prompt: str) -> str:
        img = Image.open(io.BytesIO(image_data))
        stroke_mask, core_stroke, dist_from_bg = self.extract_binary_masks(img)
        left_ports, right_ports, inverted_centers = self.locate_ports(
            stroke_mask, dist_from_bg
        )
        # Right port labels are split from symbol_pool before waypoint_pool in src/captcha.rs,
        # so badges never reuse right port characters.
        badges = self.find_all_badges(
            stroke_mask,
            dist_from_bg,
            inverted_centers,
            excluded_chars=set(right_ports.keys()),
        )
        wire_mask = self._build_wire_mask(core_stroke, badges, left_ports, right_ports)
        queried_wires, badges_per_wire = self.parse_challenge_prompt(prompt)

        answer_parts = []
        for q_wire in queried_wires:
            start_pt = left_ports.get(q_wire)
            if not start_pt:
                idx = int(q_wire) - 1 if q_wire.isdigit() else 0
                start_pt = (32.0, 32.0 + idx * 42.0)

            digits, dest_port = self.trace_wire(
                start_pt, wire_mask, badges, right_ports, badges_per_wire
            )

            # Pad with most likely near badges if trace lost connection
            while len(digits) < badges_per_wire:
                digits.append("0")

            answer_parts.extend(digits)
            answer_parts.append(dest_port)

        return "".join(answer_parts)


def run_single_attempt():
    base_url = os.environ.get("CAPTCHA_URL", "http://127.0.0.1:3000")
    font_path = os.environ.get("FONT_PATH", "assets/minimal.ttf")

    session = requests.Session()
    hardness = os.environ.get("CAPTCHA_HARDNESS")
    url = f"{base_url}/api/captcha/challenge"
    if hardness:
        url += f"?hardness={hardness}"

    try:
        challenge_resp = session.get(url).json()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server at {base_url}: {e}")
        sys.exit(1)

    prompt = challenge_resp["prompt"]
    token = challenge_resp["token"]
    server_debug_answer = challenge_resp.get("debug_correct_answer")

    image_b64 = challenge_resp["image_data_url"].split(",")[1]
    image_bytes = base64.b64decode(image_b64)

    # Save CAPTCHA image to Server Truth-encoded filename
    safe_truth = re.sub(r"[^A-Za-z0-9_-]", "", server_debug_answer or "unknown")
    output_filename = f"answer_{safe_truth}.webp"
    with open(output_filename, "wb") as f:
        f.write(image_bytes)

    # Solve once using image analysis
    solver = BezierCaptchaSolver(font_path=font_path)
    predicted_answer = solver.solve(image_bytes, prompt)

    print("--- Single Attempt Execution ---")
    print(f"Prompt           : {prompt}")
    print(f"Saved Image      : {output_filename}")
    print(f"Predicted Answer : {predicted_answer} (Length: {len(predicted_answer)})")
    if server_debug_answer:
        print(f"Server Truth     : {server_debug_answer} (Ignored by solver)")

    verify_resp = session.post(
        f"{base_url}/api/captcha/verify",
        json={"token": token, "answer": predicted_answer},
    )

    print(f"HTTP Status      : {verify_resp.status_code}")
    print(f"Server Response  : {verify_resp.json()}")

    if verify_resp.status_code == 200 and verify_resp.json().get("valid") is True:
        print("RESULT: SUCCESS - Solver cracked the CAPTCHA in 1 attempt.")
        sys.exit(0)
    else:
        print("RESULT: FAILED - Incorrect answer or rejected by server.")
        sys.exit(2)


if __name__ == "__main__":
    run_single_attempt()
