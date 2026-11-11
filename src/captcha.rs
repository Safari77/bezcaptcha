#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
use std::{f32::consts::PI, fmt, str::FromStr};

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{Rgb, RgbImage};
use imageproc::{
    drawing::{draw_antialiased_line_segment_mut, draw_filled_circle_mut},
    pixelops::interpolate,
};
use rand::{Rng, RngExt, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use zenwebp::{EncodeRequest, LosslessConfig, PixelLayout};

// === Configurable tuning constants for Hardness::Hard ===
// Adjusting these constants automatically tunes vertical port padding, decoy wire count,
// longitudinal badge zones, decoy arc numbers, prompt phrasing, and answer length.

/// Number of genuine left-side origin ports generated for [`Hardness::Hard`].
///
/// Supported range: `4..=6` (with remaining capacity up to 8 used for decoy ports).
pub const HARD_LEFT_PORTS: usize = 4; // Supported: 4..=6 real left-side options (plus decoy left ports up to 8)

/// Number of right-side destination ports (labeled `A..=H`) generated for [`Hardness::Hard`].
///
/// Supported range: `6..=8`.
pub const HARD_RIGHT_PORTS: usize = 6; // Supported: 6..=8 right-side options (A..=H)

/// Number of sequential alphanumeric badges placed along each traced wire for [`Hardness::Hard`].
///
/// Supported range: `1..=3`.
pub const HARD_BADGES_PER_WIRE: usize = 3; // Supported: 1..=3 sequential digits along each bezier

/// Number of wires the user is prompted to trace for [`Hardness::Hard`].
///
/// Supported range: `2..=3` (must not exceed [`HARD_LEFT_PORTS`]).
pub const HARD_QUERIED_WIRES: usize = 2; // Supported: 2..=3 wires to trace

// 0..10 random value is added to these:

/// Base number of background decoy micro-strokes rendered for [`Hardness::Easy`].
///
/// A random jitter of `0..10` is added at runtime.
pub const EASY_BG_MICRO_STROKES: usize = 24; // Background micro-strokes for Hardness::Easy

/// Base number of background decoy micro-strokes rendered for [`Hardness::Medium`].
///
/// A random jitter of `0..10` is added at runtime.
pub const MEDIUM_BG_MICRO_STROKES: usize = 40; // Background micro-strokes for Hardness::Medium

/// Base number of background decoy micro-strokes rendered for [`Hardness::Hard`].
///
/// A random jitter of `0..10` is added at runtime.
pub const HARD_BG_MICRO_STROKES: usize = 70; // Background micro-strokes for Hardness::Hard

// === Configurable tuning constants for underpass swerves & background micro-strokes (decoys) ===
const MAX_SWERVE_AMP: f32 = 8.0; // Peak dipole scale (actual px shift is ~0.51 * MAX_SWERVE_AMP)
const BG_MICRO_STROKE_LEN_SCALE: f32 = 1.25; // Scales stroke step lengths (e.g. 0.5..=2.5)

// Base32 (geohash) alphabet: 0-9 and b..z minus a/i/l/o.
// Case is chosen per character for maximum human legibility under affine shear and rotation:
// - Uppercase for characters with clearer cap-height geometry or open counters (B, C, F, G, J, K, P, S, V, W, X, Z)
// - Lowercase for characters whose ascenders/descenders prevent digit/letter confusion (d, e, h, m, n, q, r, t, u, y)
const BASE32_ALPHABET: [&str; 32] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "B", "C", "d", "e", "F", "G", "h", "J", "K",
    "m", "n", "P", "q", "r", "S", "t", "u", "V", "W", "X", "y", "Z",
];

const LEFT_PORT_LABELS: [&str; 8] = ["1", "2", "3", "4", "5", "6", "7", "8"];

const _: () = assert!(HARD_LEFT_PORTS >= 3 && HARD_LEFT_PORTS <= 6);
const _: () = assert!(HARD_RIGHT_PORTS >= 4 && HARD_RIGHT_PORTS <= 8);
const _: () = assert!(HARD_BADGES_PER_WIRE >= 1 && HARD_BADGES_PER_WIRE <= 3);
const _: () = assert!(HARD_QUERIED_WIRES >= 2 && HARD_QUERIED_WIRES <= 3);
const _: () = assert!(HARD_QUERIED_WIRES <= HARD_LEFT_PORTS);

/// Configurable difficulty levels for the path-tracing CAPTCHA
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Hardness {
    /// Easiest level: 4 real wires (0 decoys) connecting into 4 right ports, 1 dead-end wire,
    /// 1 badge per wire, and 2 queried wires.
    Easy,
    /// Intermediate level: 3 real wires and 3 decoy wires connecting into 6 right ports,
    /// 2 dead-end wires, 2 badges per wire, and 2 queried wires.
    Medium,
    /// Hardest level: dynamically configured via `HARD_*` tuning constants (e.g., [`HARD_LEFT_PORTS`],
    /// [`HARD_RIGHT_PORTS`], [`HARD_BADGES_PER_WIRE`], and [`HARD_QUERIED_WIRES`]).
    #[default]
    Hard,
}

impl Hardness {
    /// Returns `(real_wires, decoy_wires, total_right_ports)` for this hardness level
    pub const fn params(self) -> (usize, usize, usize) {
        match self {
            Self::Easy => (4, 0, 4),
            Self::Medium => (3, 3, 6),
            Self::Hard => {
                let decoys = if HARD_RIGHT_PORTS > HARD_LEFT_PORTS {
                    HARD_RIGHT_PORTS - HARD_LEFT_PORTS
                } else {
                    2
                };
                (HARD_LEFT_PORTS, decoys, HARD_RIGHT_PORTS)
            }
        }
    }

    /// Returns the number of dead-end wires originating from valid/decoy left ports that terminate mid-canvas
    pub const fn dead_end_wires(self) -> usize {
        match self {
            Self::Easy => 1,
            Self::Medium => 2,
            Self::Hard => 2,
        }
    }

    /// Returns the number of background curved micro-strokes for this hardness level
    pub const fn bg_micro_strokes(self) -> usize {
        match self {
            Self::Easy => EASY_BG_MICRO_STROKES,
            Self::Medium => MEDIUM_BG_MICRO_STROKES,
            Self::Hard => HARD_BG_MICRO_STROKES,
        }
    }

    /// Returns `(queried_wires_count, badges_per_wire, extra_arc_decoy_badges)`
    pub const fn query_config(self) -> (usize, usize, usize) {
        match self {
            Self::Easy => (2, 1, 2),
            Self::Medium => (2, 2, 8),
            Self::Hard => {
                let (real, decoys, _) = self.params();
                let on_wire_badges = (real + decoys) * HARD_BADGES_PER_WIRE;
                // Auto-tune extra decoy numbers on random arcs so the field stays rich without crowding
                let extra = if on_wire_badges <= 8 {
                    12
                } else if on_wire_badges <= 12 {
                    10
                } else {
                    8
                };
                (HARD_QUERIED_WIRES, HARD_BADGES_PER_WIRE, extra)
            }
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Easy => "easy",
            Self::Medium => "medium",
            Self::Hard => "hard",
        }
    }
}

impl fmt::Display for Hardness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Easy => write!(f, "easy (4 wires into 4 right ports, 0 decoys)"),
            Self::Medium => {
                write!(f, "medium (3 wires into 6 right ports, 3 decoys, 2 badges/wire)")
            }
            Self::Hard => {
                let (real, decoys, right) = self.params();
                let (queried, badges_per_wire, _) = self.query_config();
                write!(
                    f,
                    "hard ({} wires into {} right ports, {} decoys, {} queried, {} badges/wire)",
                    real, right, decoys, queried, badges_per_wire
                )
            }
        }
    }
}

impl FromStr for Hardness {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "easy" | "1" | "e" | "low" => Ok(Self::Easy),
            "medium" | "med" | "2" | "m" | "normal" => Ok(Self::Medium),
            "hard" | "3" | "h" | "high" => Ok(Self::Hard),
            _ => Err("Invalid hardness level (expected: easy/1, medium/2, or hard/3)"),
        }
    }
}

#[derive(Clone)]
pub struct CaptchaChallenge {
    pub webp_bytes: Vec<u8>,
    pub prompt: String,
    pub correct_answer: String,
}

#[derive(Clone, Copy)]
struct SamplePoint {
    x: f32,
    y: f32,
    nx: f32, // Unit normal X component (-dy/ds)
    ny: f32, // Unit normal Y component ( dx/ds)
    arc_len: f32,
}

// Lightweight stack-allocated coarse sample for fast O(1)-allocation rejection sampling
const COARSE_STEPS: usize = 48;

#[derive(Clone, Copy, Default)]
struct CoarsePoint {
    x: f32,
    y: f32,
    tx: f32, // Unit tangent X
    ty: f32, // Unit tangent Y
}

#[derive(Clone, Copy)]
struct BadgePlacement {
    x: i32,
    y: i32,
    tx: f32,                 // Local wire unit tangent X component (dy/ds = ny)
    ty: f32,                 // Local wire unit tangent Y component (-dx/ds = -nx)
    wire_idx: Option<usize>, // Some(i) for wire badges, None for decoy arc badges
    digit: &'static str,
}

#[inline(always)]
fn dist_sq(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x1 - x2;
    let dy = y1 - y2;
    dx * dx + dy * dy
}

#[inline(always)]
fn fast_len(dx: f32, dy: f32) -> f32 {
    (dx * dx + dy * dy).sqrt()
}

// Continuous stroke weight modulation parameters along a curve's arc length
#[derive(Clone, Copy)]
struct StrokeModulation {
    base_radius: f32,
    amp1: f32,
    freq1: f32,
    phase1: f32,
    amp2: f32,
    freq2: f32,
    phase2: f32,
}

impl StrokeModulation {
    fn random<R: Rng>(rng: &mut R) -> Self {
        Self {
            base_radius: rng.random_range(0.95..1.25),
            amp1: rng.random_range(0.30..0.48),
            freq1: rng.random_range(0.045..0.085),
            phase1: rng.random_range(0.0..2.0 * PI),
            amp2: rng.random_range(0.15..0.28),
            freq2: rng.random_range(0.13..0.22),
            phase2: rng.random_range(0.0..2.0 * PI),
        }
    }

    #[inline(always)]
    fn radius_at(&self, s: f32) -> f32 {
        let w1 = self.amp1 * (self.freq1 * s + self.phase1).sin();
        let w2 = self.amp2 * (self.freq2 * s + self.phase2).cos();
        (self.base_radius + w1 + w2).clamp(0.45, 1.85)
    }
}

// Draws a variable-thickness antialiased segment using normal-offset sub-strokes
// so stroke weight modulates smoothly along the curve instead of fixed 1px/3x3 segments.
fn draw_modulated_segment_mut(
    img: &mut RgbImage,
    p0: (f32, f32),
    p1: (f32, f32),
    nx: f32,
    ny: f32,
    radius: f32,
    color: Rgb<u8>,
) {
    let r = radius.clamp(0.4, 2.0);
    // Center core stroke
    draw_antialiased_line_segment_mut(
        img,
        (p0.0.round() as i32, p0.1.round() as i32),
        (p1.0.round() as i32, p1.1.round() as i32),
        color,
        interpolate,
    );

    // Continually modulate lateral offsets along the local normal vector
    let steps = (r * 2.0).ceil() as i32;
    for k in 1..=steps {
        let frac = (k as f32) / (steps as f32).max(1.0);
        let offset = r * frac;
        for sign in [-1.0_f32, 1.0_f32] {
            let ox = nx * offset * sign;
            let oy = ny * offset * sign;
            draw_antialiased_line_segment_mut(
                img,
                ((p0.0 + ox).round() as i32, (p0.1 + oy).round() as i32),
                ((p1.0 + ox).round() as i32, (p1.1 + oy).round() as i32),
                color,
                interpolate,
            );
        }
    }
}

// Renders a single glyph with continuous scale jitter, affine shear (+/- 15 deg),
// and subtle rotation (+/- 10 deg relative to a damped tangent angle) using inverse-mapped bilinear sampling.
fn draw_distorted_glyph_mut<R: Rng>(
    img: &mut RgbImage,
    font: &FontRef<'_>,
    text: &str,
    center_x: i32,
    center_y: i32,
    base_px_scale: f32,
    tangent_angle: f32,
    fg_color: Rgb<u8>,
    rng: &mut R,
) {
    let Some(ch) = text.chars().next() else {
        return;
    };

    // Keep scale variation tight and biased slightly upward so small lowercase glyphs stay legible
    let scale_jitter = rng.random_range(0.95..1.12);
    let scaled_font = font.as_scaled(PxScale::from(base_px_scale * scale_jitter));
    let glyph_id = scaled_font.glyph_id(ch);
    let glyph = glyph_id.with_scale(PxScale::from(base_px_scale * scale_jitter));

    let Some(outlined) = font.outline_glyph(glyph) else {
        return;
    };
    let bounds = outlined.px_bounds();
    let gw = (bounds.width().ceil() as i32).max(1);
    let gh = (bounds.height().ceil() as i32).max(1);

    let mut alpha_buf = vec![0.0_f32; (gw * gh) as usize];
    outlined.draw(|x, y, c| {
        let idx = (y as i32 * gw + x as i32) as usize;
        if let Some(slot) = alpha_buf.get_mut(idx) {
            *slot = c.clamp(0.0, 1.0);
        }
    });

    // Keep text upright for human readability while gently hinting the wire tangent
    let mut base_ang = tangent_angle;
    if base_ang > PI * 0.5 {
        base_ang -= PI;
    } else if base_ang < -PI * 0.5 {
        base_ang += PI;
    }

    // Apply mild random rotation (+/- 10 deg) and heavily damped tangent influence
    let rot_deg = rng.random_range(-10.0_f32..=10.0_f32);
    let theta = (base_ang * 0.22).clamp(-0.18, 0.18) + rot_deg.to_radians();
    let (sin_t, cos_t) = theta.sin_cos();

    // Apply random glyph shearing (+/- 15 deg horizontal affine shear, minimal vertical shear)
    let shear_x = rng.random_range(-15.0_f32..=15.0_f32).to_radians().tan();
    let shear_y = rng.random_range(-5.0_f32..=5.0_f32).to_radians().tan();

    // Forward affine matrix M = R(theta) * Shear(shx, shy)
    // [ m00  m01 ] = [ cos_t  -sin_t ] [ 1      shear_x ]
    // [ m10  m11 ]   [ sin_t   cos_t ] [ shear_y  1     ]
    let m00 = cos_t - sin_t * shear_y;
    let m01 = cos_t * shear_x - sin_t;
    let m10 = sin_t + cos_t * shear_y;
    let m11 = sin_t * shear_x + cos_t;

    let det = m00 * m11 - m01 * m10;
    if det.abs() < 1e-4 {
        return;
    }
    let inv_det = 1.0 / det;
    let inv00 = m11 * inv_det;
    let inv01 = -m01 * inv_det;
    let inv10 = -m10 * inv_det;
    let inv11 = m00 * inv_det;

    let half_w = 0.5 * (gw as f32);
    let half_h = 0.5 * (gh as f32);
    let pad = (half_w.max(half_h) * 1.65).ceil() as i32 + 2;
    let (img_w, img_h) = (img.width() as i32, img.height() as i32);

    // Light glyphs on a dark circle require gamma boost to prevent dark background bleed from thinning strokes
    let is_light_fg = fg_color.0[0] > 128;

    for dy in -pad..=pad {
        for dx in -pad..=pad {
            let px = center_x + dx;
            let py = center_y + dy;
            if px < 0 || px >= img_w || py < 0 || py >= img_h {
                continue;
            }

            // Map destination offset (dx, dy) back into source glyph space (sx, sy)
            let rx = dx as f32;
            let ry = dy as f32;
            let sx = inv00 * rx + inv01 * ry + half_w;
            let sy = inv10 * rx + inv11 * ry + half_h;

            if sx < 0.0 || sx >= (gw - 1) as f32 || sy < 0.0 || sy >= (gh - 1) as f32 {
                continue;
            }

            // Bilinear interpolation of glyph coverage
            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let fx = sx - (x0 as f32);
            let fy = sy - (y0 as f32);

            let c00 = alpha_buf.get((y0 * gw + x0) as usize).copied().unwrap_or(0.0);
            let c10 = alpha_buf.get((y0 * gw + (x0 + 1)) as usize).copied().unwrap_or(0.0);
            let c01 = alpha_buf.get(((y0 + 1) * gw + x0) as usize).copied().unwrap_or(0.0);
            let c11 = alpha_buf.get(((y0 + 1) * gw + (x0 + 1)) as usize).copied().unwrap_or(0.0);

            let c0 = c00 + fx * (c10 - c00);
            let c1 = c01 + fx * (c11 - c01);
            let raw_cov = (c0 + fy * (c1 - c0)).clamp(0.0, 1.0);

            // Boost coverage curve so thin features (like lowercase 'r') stay bold, especially on dark halos
            let coverage = if is_light_fg {
                (raw_cov.powf(0.62) * 1.25).clamp(0.0, 1.0)
            } else {
                (raw_cov.powf(0.82) * 1.10).clamp(0.0, 1.0)
            };

            if coverage > 0.02 {
                let dst = *img.get_pixel(px as u32, py as u32);
                let blended = interpolate(fg_color, dst, coverage);
                img.put_pixel(px as u32, py as u32, blended);
            }
        }
    }
}

// Multi-frequency Fourier parameters for continuous normal-vector orientation jitter
struct HarmonicJitter {
    amps: [f32; 3],
    freqs: [f32; 3],
    phases: [f32; 3],
}

impl HarmonicJitter {
    fn random<R: Rng>(rng: &mut R) -> Self {
        Self {
            // Amplitudes kept small (1.0px to 2.8px) so macroscopic shape stays readable
            amps: [
                rng.random_range(2.0..2.8),
                rng.random_range(1.4..2.1),
                rng.random_range(1.0..1.6),
            ],
            // Incommensurate spatial frequencies (wavelengths ~75px, ~38px, and ~22px)
            freqs: [
                rng.random_range(0.075..0.095),
                rng.random_range(0.155..0.185),
                rng.random_range(0.265..0.310),
            ],
            phases: [
                rng.random_range(0.0..2.0 * PI),
                rng.random_range(0.0..2.0 * PI),
                rng.random_range(0.0..2.0 * PI),
            ],
        }
    }

    // Evaluates normal displacement at arc length s along a wire of total length total_len
    fn eval(&self, s: f32, total_len: f32) -> f32 {
        if total_len <= 1.0 {
            return 0.0;
        }
        // Smooth sin^2 window pins jitter to 0.0 at both port endpoints (s = 0 and s = total_len)
        let envelope = (PI * (s / total_len).clamp(0.0, 1.0)).sin().powi(2);
        let [a0, a1, a2] = self.amps;
        let [f0, f1, f2] = self.freqs;
        let [p0, p1, p2] = self.phases;
        let wave = a0 * (f0 * s + p0).sin() + a1 * (f1 * s + p1).sin() + a2 * (f2 * s + p2).sin();
        envelope * wave
    }
}

// Generates randomized, non-overlapping port coordinates in top-to-bottom order.
// Varies both the vertical gaps between adjacent ports and the horizontal X alignment
// so a solver cannot rely on fixed grid coordinates or uniform inter-port spacing.
fn random_port_coords<R: Rng>(
    count: usize,
    height: f32,
    v_pad: f32,
    base_x: f32,
    x_jitter: f32,
    rng: &mut R,
) -> Vec<(f32, f32)> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        let x = base_x + rng.random_range(-x_jitter..=x_jitter);
        let y = height * 0.5 + rng.random_range(-25.0..=25.0);
        return vec![(x, y)];
    }

    let avail_h = (height - 2.0 * v_pad).max(120.0);
    // Guarantee enough vertical clearance that 20px font glyphs never collide
    let min_gap = (avail_h / (count as f32 + 1.15)).clamp(20.0, 30.0);
    let slack = (avail_h - min_gap * (count as f32 - 1.0)).max(4.0);

    // Distribute slack across top margin, (count - 1) inter-port gaps, and bottom margin
    let mut weights = vec![0.0_f32; count + 1];
    for w in weights.iter_mut() {
        *w = rng.random_range(0.12..1.0);
    }
    // Give one random inter-port gap extra weight to create asymmetric spacing
    let wide_slot = rng.random_range(0..=count);
    if let Some(w) = weights.get_mut(wide_slot) {
        *w += rng.random_range(0.8..1.8);
    }

    let weight_sum: f32 = weights.iter().sum::<f32>().max(1e-5);
    let mut coords = Vec::with_capacity(count);
    let first_weight = weights.first().copied().unwrap_or(0.5);
    let mut curr_y = v_pad + slack * (first_weight / weight_sum);

    for (i, &weight) in weights.iter().enumerate().take(count) {
        if i > 0 {
            curr_y += min_gap + slack * (weight / weight_sum);
        }
        let x = base_x + rng.random_range(-x_jitter..=x_jitter);
        coords.push((x, curr_y));
    }

    coords
}

// Evaluates a cubic Bezier curve at step t in [0.0, 1.0]
#[inline(always)]
fn cubic_bezier(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    t: f32,
) -> (f32, f32) {
    let u = 1.0 - t;
    let tt = t * t;
    let uu = u * u;
    let uuu = uu * u;
    let ttt = tt * t;

    let x = uuu * p0.0 + 3.0 * uu * t * p1.0 + 3.0 * u * tt * p2.0 + ttt * p3.0;
    let y = uuu * p0.1 + 3.0 * uu * t * p1.1 + 3.0 * u * tt * p2.1 + ttt * p3.1;
    (x, y)
}

// Evaluates the first derivative dB/dt of a cubic Bezier curve to derive the unit normal vector
#[inline(always)]
fn cubic_bezier_derivative(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    t: f32,
) -> (f32, f32) {
    let u = 1.0 - t;
    let dx =
        3.0 * u * u * (p1.0 - p0.0) + 6.0 * u * t * (p2.0 - p1.0) + 3.0 * t * t * (p3.0 - p2.0);
    let dy =
        3.0 * u * u * (p1.1 - p0.1) + 6.0 * u * t * (p2.1 - p1.1) + 3.0 * t * t * (p3.1 - p2.1);
    (dx, dy)
}

// Evaluates 49 coarse points and unit tangents on the stack in < 500ns (no trig, no heap allocation)
fn eval_coarse_bezier(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
) -> [CoarsePoint; COARSE_STEPS + 1] {
    let inv = 1.0 / (COARSE_STEPS as f32);
    std::array::from_fn(|i| {
        let t = (i as f32) * inv;
        let (x, y) = cubic_bezier(p0, p1, p2, p3, t);
        let (dx, dy) = cubic_bezier_derivative(p0, p1, p2, p3, t);
        let inv_len = 1.0 / fast_len(dx, dy).max(1e-5);
        CoarsePoint { x, y, tx: dx * inv_len, ty: dy * inv_len }
    })
}

// Ultra-fast O(1)-allocation clarity check on coarse Bezier curves BEFORE expensive 2000-step sampling.
// Returns (is_acceptable, penalty_score).
// Rejects curves that:
// 1. Cross or graze another wire (< 7.5px) at a shallow angle (|cos(theta)| > 0.86, ~30 deg)
// 2. Overlap or run nearly parallel within 10.5px of another wire for >= 3 coarse steps (~28px)
fn check_coarse_clarity(
    cand: &[CoarsePoint; COARSE_STEPS + 1],
    existing: &[[CoarsePoint; COARSE_STEPS + 1]],
    shared_origin_wire: Option<usize>,
    shared_dest_wire: Option<usize>,
) -> (bool, f32) {
    const CROSS_DIST_SQ: f32 = 7.5 * 7.5;
    const OVERLAP_DIST_SQ: f32 = 10.5 * 10.5;
    const MAX_SHALLOW_COS: f32 = 0.86; // ~30.6 degrees minimum crossing angle

    let mut penalty = 0.0_f32;
    let mut hard_reject = false;

    for (w_idx, other) in existing.iter().enumerate() {
        let min_step = if shared_origin_wire == Some(w_idx) {
            5 // Ignore divergence right at a shared origin port
        } else {
            1
        };
        let max_step = if shared_dest_wire == Some(w_idx) {
            COARSE_STEPS - 4 // Ignore convergence right at a shared destination port
        } else {
            COARSE_STEPS
        };

        let mut overlap_run = 0usize;

        for (ci, &cpt) in cand.iter().enumerate().take(max_step + 1).skip(min_step) {
            // Since x increases almost monotonically with step t, we only need to check a window of `oi`
            let oi_start = ci.saturating_sub(14).max(min_step);
            let oi_end = (ci + 14).min(max_step);

            let mut step_close = false;
            for &opt in other.iter().take(oi_end + 1).skip(oi_start) {
                if (cpt.x - opt.x).abs() > 11.0 || (cpt.y - opt.y).abs() > 11.0 {
                    continue;
                }
                let d2 = dist_sq(cpt.x, cpt.y, opt.x, opt.y);
                if d2 < OVERLAP_DIST_SQ {
                    step_close = true;
                    let cos_ang = (cpt.tx * opt.tx + cpt.ty * opt.ty).abs();
                    if d2 < CROSS_DIST_SQ {
                        if cos_ang > MAX_SHALLOW_COS {
                            hard_reject = true;
                            penalty += 400.0 * (cos_ang - MAX_SHALLOW_COS + 0.05);
                        } else if cos_ang > 0.72 {
                            penalty += 25.0 * cos_ang;
                        }
                    } else if cos_ang > 0.88 {
                        penalty += 15.0 * cos_ang;
                    }
                }
            }

            if step_close {
                overlap_run += 1;
                if overlap_run >= 3 {
                    hard_reject = true;
                    penalty += 200.0;
                }
            } else {
                overlap_run = 0;
            }
        }
    }

    (!hard_reject, penalty)
}

// Resamples a cubic Bezier curve into equidistant points along its true pixel arc length
// and applies harmonic normal-vector orientation jitter
fn sample_bezier_by_arc_length<R: Rng>(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    step_px: f32,
    rng: &mut R,
) -> Vec<SamplePoint> {
    const FINE_STEPS: usize = 2000;
    let mut samples = Vec::with_capacity(600);
    let mut prev = p0;
    let mut total_len = 0.0;
    let mut next_emit = 0.0;

    let (dx0, dy0) = cubic_bezier_derivative(p0, p1, p2, p3, 0.0);
    let norm0 = fast_len(dx0, dy0).max(1e-5);
    samples.push(SamplePoint { x: p0.0, y: p0.1, nx: -dy0 / norm0, ny: dx0 / norm0, arc_len: 0.0 });

    for i in 1..=FINE_STEPS {
        let t = i as f32 / FINE_STEPS as f32;
        let curr = cubic_bezier(p0, p1, p2, p3, t);
        let seg_len = fast_len(curr.0 - prev.0, curr.1 - prev.1);
        total_len += seg_len;

        if total_len >= next_emit {
            let (dx, dy) = cubic_bezier_derivative(p0, p1, p2, p3, t);
            let norm = fast_len(dx, dy).max(1e-5);
            samples.push(SamplePoint {
                x: curr.0,
                y: curr.1,
                nx: -dy / norm,
                ny: dx / norm,
                arc_len: total_len,
            });
            next_emit += step_px;
        }
        prev = curr;
    }

    // Apply continuous multi-frequency normal perturbation
    let jitter = HarmonicJitter::random(rng);
    for pt in samples.iter_mut() {
        let offset = jitter.eval(pt.arc_len, total_len);
        pt.x += pt.nx * offset;
        pt.y += pt.ny * offset;
    }

    recompute_arc_lengths(&mut samples);
    samples
}

// Recomputes cumulative Euclidean arc length and finite-difference unit normals after displacements
fn recompute_arc_lengths(samples: &mut [SamplePoint]) {
    let Some((first, rest)) = samples.split_first_mut() else {
        return;
    };
    first.arc_len = 0.0;
    let mut prev_x = first.x;
    let mut prev_y = first.y;
    let mut acc = 0.0;

    for pt in rest.iter_mut() {
        let dx = pt.x - prev_x;
        let dy = pt.y - prev_y;
        let seg = fast_len(dx, dy);
        acc += seg;
        pt.arc_len = acc;
        if seg > 1e-4 {
            pt.nx = -dy / seg;
            pt.ny = dx / seg;
        }
        prev_x = pt.x;
        prev_y = pt.y;
    }

    if let Some(second) = rest.first() {
        first.nx = second.nx;
        first.ny = second.ny;
    }
}

// Injects an antisymmetric S-wave deflection centered on each underpass cut so pre-cut tangents miss the exit,
// scaled by crossing angle sin^2(theta) so shallow crossings are never shifted into ambiguous alignment.
fn apply_underpass_swerves(all_wires: &mut [Vec<SamplePoint>], cut_radius: f32) {
    // Scale spatial spread with amplitude so high MAX_SWERVE_AMP never folds/cusps the wire
    let swerve_sigma: f32 = (7.0_f32).max(MAX_SWERVE_AMP * 0.85);
    let cut_radius_sq = cut_radius * cut_radius;
    let min_swerve_sep = (swerve_sigma * 3.2).max(26.0);

    let mut swerve_centers: Vec<(usize, f32, f32)> = Vec::new();

    for (i, wire) in all_wires.iter().enumerate() {
        let mut last_recorded_s = -1000.0;
        for pt in wire {
            if pt.arc_len - last_recorded_s < min_swerve_sep {
                continue;
            }
            for (j, other_wire) in all_wires.iter().enumerate() {
                if i == j {
                    continue;
                }
                let mut crossing_cos: Option<f32> = None;
                for other_pt in other_wire.iter().step_by(2) {
                    if (pt.x - other_pt.x).abs() >= cut_radius
                        || (pt.y - other_pt.y).abs() >= cut_radius
                    {
                        continue;
                    }
                    if dist_sq(pt.x, pt.y, other_pt.x, other_pt.y) < cut_radius_sq {
                        let mid_x = 0.5 * (pt.x + other_pt.x);
                        if wire_elevation(i, mid_x) < wire_elevation(j, mid_x) {
                            crossing_cos = Some((pt.nx * other_pt.nx + pt.ny * other_pt.ny).abs());
                            break;
                        }
                    }
                }
                if let Some(cos_ang) = crossing_cos {
                    let sin_sq = (1.0 - cos_ang * cos_ang).clamp(0.0, 1.0);
                    if sin_sq > 0.25 {
                        let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                        swerve_centers.push((i, pt.arc_len, sign * MAX_SWERVE_AMP * sin_sq));
                        last_recorded_s = pt.arc_len;
                    }
                    break;
                }
            }
        }
    }

    // Apply all swerves for a wire in a single pass so s_center indices and normals stay coherent
    for (wire_idx, wire) in all_wires.iter_mut().enumerate() {
        let total_len = wire.last().map(|p| p.arc_len).unwrap_or(1.0);
        let mut modified = false;

        for pt in wire.iter_mut() {
            let mut total_delta = 0.0_f32;
            for &(w_idx, s_center, signed_amp) in &swerve_centers {
                if w_idx != wire_idx {
                    continue;
                }
                let ds = pt.arc_len - s_center;
                if ds.abs() < swerve_sigma * 3.2 {
                    let u = ds / swerve_sigma;
                    // Fast algebraic approximation to u * exp(-0.5 * u^2): u / (1 + 0.5 * u^2)^2
                    let denom = 1.0 + 0.5 * u * u;
                    let dipole = signed_amp * u / (denom * denom);
                    let endpoint_guard = (PI * (pt.arc_len / total_len).clamp(0.0, 1.0)).sin();
                    total_delta += dipole * endpoint_guard;
                }
            }
            if total_delta.abs() > 1e-5 {
                pt.x += pt.nx * total_delta;
                pt.y += pt.ny * total_delta;
                modified = true;
            }
        }

        if modified {
            recompute_arc_lengths(wire);
        }
    }
}

// Smooth 3D elevation wave so wires weave over and under each other rather than stacking flat
#[inline(always)]
fn wire_elevation(wire_idx: usize, x: f32) -> f32 {
    (x * 0.038 + (wire_idx as f32) * 1.73).sin()
}

// Asymmetric dash-dot-dash cadence (period = 34px) with spatial chirp modulation to defeat fixed-stride Hough/Radon transforms
#[inline(always)]
fn is_dash_on(arc_len: f32, phase_offset: f32) -> bool {
    let warped_s = arc_len + 9.5 * (arc_len * 0.021 + phase_offset * 0.18).sin();
    let pos = (warped_s + phase_offset).rem_euclid(34.0);
    (pos < 14.0) || (19.0..28.0).contains(&pos)
}

// Picks a character from the shuffled pool, consuming each unique character once before wrapping.
// Because `pool` has 32 unique characters and right ports + wire badges <= 32, every character
// in the correct answer is guaranteed to be unique.
fn pick_non_clashing_digit<R: Rng>(
    x: f32,
    y: f32,
    existing: &[BadgePlacement],
    pool: &[&'static str],
    cursor: &mut usize,
    rng: &mut R,
) -> &'static str {
    const MIN_SAME_DIGIT_DIST_SQ: f32 = 52.0 * 52.0;
    let pool_len = pool.len();
    if pool_len == 0 {
        return "0";
    }
    if let Some(&unique_char) = pool.get(*cursor) {
        *cursor += 1;
        return unique_char;
    }
    for attempt in 0..pool_len {
        let Some(&candidate) = pool.get((*cursor + attempt) % pool_len) else {
            continue;
        };
        let clash = existing.iter().any(|b| {
            b.digit == candidate && dist_sq(x, y, b.x as f32, b.y as f32) < MIN_SAME_DIGIT_DIST_SQ
        });
        if !clash {
            *cursor += attempt + 1;
            return candidate;
        }
    }
    let fallback = pool.get(rng.random_range(0..pool_len)).copied().unwrap_or("0");
    *cursor += 1;
    fallback
}

// A decoy badge must not touch another wire. Besides making ownership ambiguous,
// its 8.5px halo would otherwise cut a gap in that wire as well.
fn decoy_site_is_clear(
    all_wires: &[Vec<SamplePoint>],
    own_wire_idx: usize,
    pt: &SamplePoint,
) -> bool {
    // Slightly more than the 17.5px clearance already used for arc decoys.
    const MIN_FOREIGN_WIRE_D2: f32 = 18.0 * 18.0;

    all_wires.iter().enumerate().all(|(other_idx, other_wire)| {
        other_idx == own_wire_idx
            || other_wire
                .iter()
                .step_by(3)
                .all(|other_pt| dist_sq(pt.x, pt.y, other_pt.x, other_pt.y) >= MIN_FOREIGN_WIRE_D2)
    })
}

// Places `badges_per_wire` sequential waypoint digits along each active wire across longitudinal bands.
// Randomizes horizontal positions within each band so queried wire badges are never biased
// toward the leftmost slots, and centers all wire badges directly on their wire.
fn select_wire_badges<R: Rng>(
    all_wires: &[Vec<SamplePoint>],
    badged_wire_count: usize,
    queried_wires: &[usize],
    badges_per_wire: usize,
    pool: &[&'static str],
    pool_cursor: &mut usize,
    rng: &mut R,
) -> (Vec<Vec<BadgePlacement>>, Vec<BadgePlacement>) {
    let total_wires = badged_wire_count.min(all_wires.len());
    let mut per_wire: Vec<Vec<BadgePlacement>> =
        vec![Vec::with_capacity(badges_per_wire); all_wires.len()];
    let mut all_placed: Vec<BadgePlacement> = Vec::with_capacity(total_wires * badges_per_wire);

    // Prioritize queried wires for clean spacing, but randomize their longitudinal target frac
    let mut order = Vec::with_capacity(total_wires);
    order.extend_from_slice(queried_wires);
    for idx in 0..total_wires {
        if !queried_wires.contains(&idx) {
            order.push(idx);
        }
    }

    let band_span = 0.68 / (badges_per_wire as f32);

    for band in 0..badges_per_wire {
        let band_min = 0.16 + (band as f32) * band_span;
        let band_max = band_min + band_span;

        // Shuffle horizontal target fractions inside this band so queried wires don't always sit left
        let mut fracs: Vec<f32> = (0..total_wires)
            .map(|slot| {
                let t = (slot as f32 + rng.random_range(0.15..0.85)) / (total_wires as f32);
                band_min + 0.03 + t * (band_span - 0.06)
            })
            .collect();
        fracs.shuffle(rng);

        for (&idx, &raw_frac) in order.iter().zip(fracs.iter()) {
            let Some(wire) = all_wires.get(idx) else {
                continue;
            };
            let is_queried = queried_wires.contains(&idx);
            let frac = raw_frac.clamp(band_min + 0.02, band_max - 0.02);

            let min_idx = ((wire.len() as f32 * band_min) as usize).max(32);
            let max_idx =
                ((wire.len() as f32 * band_max) as usize).min(wire.len().saturating_sub(32));
            let center_idx = ((wire.len() as f32 * frac) as usize).clamp(min_idx, max_idx);
            let search_radius = (58 / badges_per_wire).max(24);
            let start_idx = center_idx.saturating_sub(search_radius).max(min_idx);
            let end_idx = (center_idx + search_radius).min(max_idx);

            let mut best_pt: Option<SamplePoint> = None;
            let mut best_score = f32::NEG_INFINITY;

            for (cand_idx, &pt) in
                wire.iter().enumerate().take(end_idx + 1).skip(start_idx).step_by(6)
            {
                // For optional decoys, clearance is mandatory rather than merely
                // increasing/decreasing the candidate's score.
                if !is_queried && !decoy_site_is_clear(all_wires, idx, &pt) {
                    continue;
                }
                let mut min_wire_d2 = f32::INFINITY;
                let mut min_queried_d2 = f32::INFINITY;

                for (other_idx, other_wire) in all_wires.iter().enumerate() {
                    if idx == other_idx {
                        continue;
                    }
                    let is_other_queried = queried_wires.contains(&other_idx);
                    for other_pt in other_wire.iter().step_by(8) {
                        let d2 = dist_sq(pt.x, pt.y, other_pt.x, other_pt.y);
                        if d2 < min_wire_d2 {
                            min_wire_d2 = d2;
                        }
                        if is_other_queried && d2 < min_queried_d2 {
                            min_queried_d2 = d2;
                        }
                    }
                }

                let mut min_badge_d2 = f32::INFINITY;
                for b in &all_placed {
                    let d2 = dist_sq(pt.x, pt.y, b.x as f32, b.y as f32);
                    if d2 < min_badge_d2 {
                        min_badge_d2 = d2;
                    }
                }

                // Enforce a strong penalty if two badges come within 24px (24^2 = 576)
                let badge_term = if min_badge_d2 < 576.0 {
                    min_badge_d2 * 4.0 - 3000.0
                } else {
                    min_badge_d2.min(1600.0) * 1.5
                };

                let mut score = min_wire_d2.min(900.0) + badge_term;
                if !is_queried {
                    score += min_queried_d2.min(1225.0) * 1.6;
                }
                score -= (cand_idx as f32 - center_idx as f32).abs() * 1.2;

                if score > best_score {
                    best_score = score;
                    best_pt = Some(pt);
                }
            }

            let best_pt = match best_pt {
                Some(pt) => pt,
                // Decoys are optional: never fall back to the unchecked center
                // point when every candidate was rejected.
                None if !is_queried => continue,
                // Preserve the existing behavior for mandatory answer badges.
                None => {
                    let Some(&fallback_pt) = wire.get(center_idx).or_else(|| wire.first()) else {
                        continue;
                    };
                    fallback_pt
                }
            };

            let bx = best_pt.x.round() as i32;
            let by = best_pt.y.round() as i32;
            let digit =
                pick_non_clashing_digit(bx as f32, by as f32, &all_placed, pool, pool_cursor, rng);
            // Unit tangent vector t(s) from unit normal (-dy/ds, dx/ds) -> (dx/ds, dy/ds) = (ny, -nx)
            let placement = BadgePlacement {
                x: bx,
                y: by,
                tx: best_pt.ny,
                ty: -best_pt.nx,
                wire_idx: Some(idx),
                digit,
            };
            if let Some(wire_badges) = per_wire.get_mut(idx) {
                wire_badges.push(placement);
            }
            all_placed.push(placement);
        }
    }

    // Ensure badges on each wire are strictly ordered left-to-right by X coordinate
    for wire_badges in per_wire.iter_mut() {
        wire_badges.sort_by_key(|b| b.x);
    }

    (per_wire, all_placed)
}

// Places extra decoy numbers in open pockets (especially near the left ports and between wires)
// so a bot cannot simply pick the closest number to port 1 or 2. Each decoy number is paired
// with a curved dashed micro-stroke passing directly through its halo.
fn select_decoy_arc_badges<R: Rng>(
    all_wires: &[Vec<SamplePoint>],
    all_placed: &mut Vec<BadgePlacement>,
    extra_count: usize,
    width: u32,
    height: u32,
    padding: f32,
    pool: &[&'static str],
    pool_cursor: &mut usize,
    rng: &mut R,
) -> Vec<BadgePlacement> {
    const MIN_WIRE_CLEARANCE_SQ: f32 = 16.0 * 16.0;
    const MIN_BADGE_CLEARANCE_SQ: f32 = 22.5 * 22.5;

    let mut decoy_badges = Vec::with_capacity(extra_count);
    let min_x = padding + 22.0;
    let max_x = (width as f32) - padding - 26.0;
    let min_y = 20.0_f32;
    let max_y = (height as f32) - 20.0;

    for slot in 0..extra_count {
        // Bias the first half of decoy numbers toward the left half of the canvas
        // to trap nearest-neighbor / port-proximity heuristics
        let x_upper = if slot * 2 < extra_count {
            padding + (width as f32 - 2.0 * padding) * 0.55
        } else {
            max_x
        };

        let mut best_cand: Option<(f32, f32, f32)> = None;
        for _ in 0..45 {
            let cx = rng.random_range(min_x..x_upper.max(min_x + 20.0));
            let cy = rng.random_range(min_y..max_y);

            let badge_ok = all_placed
                .iter()
                .all(|b| dist_sq(cx, cy, b.x as f32, b.y as f32) >= MIN_BADGE_CLEARANCE_SQ);
            if !badge_ok {
                continue;
            }

            let mut min_w_d2 = f32::INFINITY;
            for wire in all_wires {
                for wpt in wire.iter().step_by(5) {
                    let d2 = dist_sq(cx, cy, wpt.x, wpt.y);
                    if d2 < min_w_d2 {
                        min_w_d2 = d2;
                    }
                }
            }

            if min_w_d2 >= MIN_WIRE_CLEARANCE_SQ {
                let score = min_w_d2.min(1600.0);
                if best_cand.is_none_or(|(_, _, s)| score > s) {
                    best_cand = Some((cx, cy, score));
                }
            }
        }

        if let Some((cx, cy, _)) = best_cand {
            let bx = cx.round() as i32;
            let by = cy.round() as i32;
            let digit =
                pick_non_clashing_digit(bx as f32, by as f32, all_placed, pool, pool_cursor, rng);
            let theta0 = rng.random_range(-0.65_f32..0.65_f32);
            let placement = BadgePlacement {
                x: bx,
                y: by,
                tx: theta0.cos(),
                ty: theta0.sin(),
                wire_idx: None,
                digit,
            };
            all_placed.push(placement);
            decoy_badges.push(placement);
        }
    }

    decoy_badges
}

// Draws the pristine cubic Béziers for the correct answer wires (before harmonic jitter
// and underpass swerves), and outlines their answer badges.
fn draw_debug_overlay(
    img: &mut RgbImage,
    clear_beziers: &[[(f32, f32); 4]],
    per_wire_badges: &[Vec<BadgePlacement>],
    queried_wires: &[usize],
) {
    const CURVE_STEPS: usize = 512;
    let red = Rgb([255, 0, 0]);

    // Draw only the Bézier curves belonging to the correct answer wires.
    for &wire_idx in queried_wires {
        let Some(&[p0, p1, p2, p3]) = clear_beziers.get(wire_idx) else {
            continue;
        };
        let mut previous = p0;

        for step in 1..=CURVE_STEPS {
            let t = step as f32 / CURVE_STEPS as f32;
            let current = cubic_bezier(p0, p1, p2, p3, t);

            draw_antialiased_line_segment_mut(
                img,
                (previous.0.round() as i32, previous.1.round() as i32),
                (current.0.round() as i32, current.1.round() as i32),
                red,
                interpolate,
            );

            previous = current;
        }
    }

    // Highlight the badges on the queried wires that form the correct answer.
    let (img_w, img_h) = (img.width() as i32, img.height() as i32);
    for &wire_idx in queried_wires {
        let Some(badges) = per_wire_badges.get(wire_idx) else {
            continue;
        };
        for badge in badges {
            // Draw a solid, gap-free 2px ring covering radius 11 to 12 (10.5 <= r <= 12.5)
            for dy in -13..=13 {
                for dx in -13..=13 {
                    let d2 = dx * dx + dy * dy;
                    // 10.5^2 = 110.25, 12.5^2 = 156.25
                    if (111..=156).contains(&d2) {
                        let px = badge.x + dx;
                        let py = badge.y + dy;
                        if px >= 0 && px < img_w && py >= 0 && py < img_h {
                            img.put_pixel(px as u32, py as u32, red);
                        }
                    }
                }
            }
        }
    }
}

pub fn generate_path_captcha(hardness: Hardness) -> Result<CaptchaChallenge, zenwebp::EncodeError> {
    generate_path_captcha_impl(hardness, false).map(|(challenge, _)| challenge)
}

/// Debugging only: adds red pristine Béziers and outlines the answer badges.
/// Returns (original_challenge, debug_overlay_challenge).
/// Do not use this variant for challenges served to users.
pub fn generate_path_captcha_with_debug_overlay(
    hardness: Hardness,
) -> Result<(CaptchaChallenge, CaptchaChallenge), zenwebp::EncodeError> {
    let (challenge, debug_challenge) = generate_path_captcha_impl(hardness, true)?;
    let debug_challenge = debug_challenge.unwrap_or_else(|| challenge.clone());
    Ok((challenge, debug_challenge))
}

pub fn generate_path_captcha_impl(
    hardness: Hardness,
    debug_overlay: bool,
) -> Result<(CaptchaChallenge, Option<CaptchaChallenge>), zenwebp::EncodeError> {
    const WIDTH: u32 = 460;
    const HEIGHT: u32 = 270;
    const PADDING: f32 = 36.0;
    const MAX_COARSE_ATTEMPTS: usize = 24;

    let (real_wires, decoy_wires, total_right_ports) = hardness.params();
    let dead_end_wires = hardness.dead_end_wires();
    let (queried_count, badges_per_wire, extra_decoy_badges) = hardness.query_config();
    let mut bg_micro_stroke_count = hardness.bg_micro_strokes();

    // Render decoy left ports (e.g. 5..=8) alongside real ports (1..=real_wires)
    // so all full-span decoy wires originate from realistic left ports at x ~ 14.
    let decoy_left_ports = decoy_wires.max(1).min(LEFT_PORT_LABELS.len() - real_wires);
    let total_left_ports = (real_wires + decoy_left_ports).min(LEFT_PORT_LABELS.len());
    let full_span_wires = real_wires + decoy_wires;
    let total_wires = full_span_wires + dead_end_wires;

    const UNDERPASS_CUT_RADIUS: f32 = 6.5;
    const UNDERPASS_CUT_RADIUS_SQ: f32 = UNDERPASS_CUT_RADIUS * UNDERPASS_CUT_RADIUS;
    const OVERPASS_SOLID_RADIUS: f32 = 9.0;
    const OVERPASS_SOLID_RADIUS_SQ: f32 = OVERPASS_SOLID_RADIUS * OVERPASS_SOLID_RADIUS;

    // Empty halo radius around each number and solid-wire entry/exit lock
    const BADGE_HALO_RADIUS: f32 = 10.5;
    const BADGE_HALO_RADIUS_SQ: f32 = BADGE_HALO_RADIUS * BADGE_HALO_RADIUS;
    const BADGE_SOLID_LOCK_RADIUS_SQ: f32 = 16.0 * 16.0;

    let bg_color = Rgb([248, 249, 250]);
    let stroke_color = Rgb([35, 35, 40]);
    let mut img = RgbImage::from_pixel(WIDTH, HEIGHT, bg_color);
    let mut rng = rand::rng();

    // Left ports include both real ports (1..=real_wires) and decoy ports (real_wires+1..=total_left_ports)
    let left_labels = LEFT_PORT_LABELS.get(..total_left_ports).unwrap_or(&LEFT_PORT_LABELS);

    // Shuffle all 32 Base32 characters; allocate right port labels first, then badges from the remaining pool
    // so right ports and wire badges never share characters and every character in correct_answer is unique.
    let mut symbol_pool: Vec<&'static str> = BASE32_ALPHABET.to_vec();
    symbol_pool.shuffle(&mut rng);

    let right_port_count = total_right_ports.min(symbol_pool.len());
    let (right_labels, waypoint_pool) = symbol_pool.split_at(right_port_count);
    let mut pool_cursor: usize = 0;

    // Shuffle right-hand destinations: indices 0..real_wires are real wires,
    // indices real_wires..full_span_wires are full-span decoy wires from decoy left ports.
    let mut dest_indices: Vec<usize> = (0..right_port_count).collect();
    dest_indices.shuffle(&mut rng);
    while dest_indices.len() < full_span_wires {
        dest_indices.push(rng.random_range(0..right_port_count.max(1)));
    }

    // Auto-tune vertical padding when port count increases and randomize port coords.
    // Shuffle the vertical slot assignment of left ports so decoy ports (e.g. 5..8) are
    // interspersed vertically with real ports (1..4) rather than always sitting at the bottom.
    let left_v_pad = if total_left_ports >= 6 { 22.0 } else { 32.0 };
    let right_v_pad = if right_port_count >= 7 { 22.0 } else { 32.0 };
    let mut left_port_coords =
        random_port_coords(total_left_ports, HEIGHT as f32, left_v_pad, 14.0, 4.0, &mut rng);
    left_port_coords.shuffle(&mut rng);

    let right_port_coords = random_port_coords(
        right_port_count,
        HEIGHT as f32,
        right_v_pad,
        (WIDTH - 24) as f32,
        4.0,
        &mut rng,
    );

    // Pick `queried_count` distinct real wires and sort them in ascending order
    let mut query_indices: Vec<usize> = (0..real_wires).collect();
    query_indices.shuffle(&mut rng);
    query_indices.truncate(queried_count);
    query_indices.sort_unstable();

    let mut all_wires: Vec<Vec<SamplePoint>> = Vec::with_capacity(total_wires);
    let mut accepted_coarse: Vec<[CoarsePoint; COARSE_STEPS + 1]> = Vec::with_capacity(total_wires);
    let mut dash_phases: Vec<f32> = Vec::with_capacity(total_wires);
    let mut wire_modulations: Vec<StrokeModulation> = Vec::with_capacity(total_wires);
    let mut clear_beziers: Vec<[(f32, f32); 4]> = Vec::with_capacity(total_wires);
    let mut wire_origin_ports: Vec<usize> = Vec::with_capacity(total_wires);

    // 1. Sample the real wires (originating from left ports 0..real_wires)
    for (port_idx, (&(lx, ly), &dest_idx)) in
        left_port_coords.iter().zip(dest_indices.iter()).take(real_wires).enumerate()
    {
        let &(rx, ry) = right_port_coords.get(dest_idx).unwrap_or(&(WIDTH as f32 - 24.0, ly));
        let p0 = (lx + 18.0, ly);
        let p3 = (rx - 8.0, ry);

        let mut best_ctrl = [p0, p0, p3, p3];
        let mut best_coarse = [CoarsePoint::default(); COARSE_STEPS + 1];
        let mut best_penalty = f32::INFINITY;

        for _ in 0..MAX_COARSE_ATTEMPTS {
            let p1 = (
                PADDING + rng.random_range(68.0..165.0),
                PADDING + rng.random_range(0.0..(HEIGHT as f32 - 2.0 * PADDING)),
            );
            let p2 = (
                WIDTH as f32 - PADDING - rng.random_range(68.0..165.0),
                PADDING + rng.random_range(0.0..(HEIGHT as f32 - 2.0 * PADDING)),
            );

            let coarse = eval_coarse_bezier(p0, p1, p2, p3);
            let (ok, penalty) = check_coarse_clarity(&coarse, &accepted_coarse, None, None);

            if penalty < best_penalty {
                best_penalty = penalty;
                best_ctrl = [p0, p1, p2, p3];
                best_coarse = coarse;
            }
            if ok && penalty < 20.0 {
                break;
            }
        }

        let [p0, p1, p2, p3] = best_ctrl;
        accepted_coarse.push(best_coarse);
        clear_beziers.push(best_ctrl);
        wire_origin_ports.push(port_idx);
        all_wires.push(sample_bezier_by_arc_length(p0, p1, p2, p3, 1.0, &mut rng));
        dash_phases.push(rng.random_range(0.0..34.0));
        wire_modulations.push(StrokeModulation::random(&mut rng));
    }

    // 2. Sample full-span decoy wires originating from decoy left ports (numbers real_wires+1..=total_left_ports)
    // and traveling all the way across the canvas into right ports.
    for (decoy_slot, (i, &dest_idx)) in
        dest_indices.iter().enumerate().take(full_span_wires).skip(real_wires).enumerate()
    {
        let port_idx = real_wires + (decoy_slot % decoy_left_ports.max(1));
        let &(lx, ly) = left_port_coords.get(port_idx).unwrap_or(&(14.0, HEIGHT as f32 * 0.5));
        let &(rx, ry) =
            right_port_coords.get(dest_idx).unwrap_or(&(WIDTH as f32 - 24.0, HEIGHT as f32 * 0.5));
        let p0 = (lx + 18.0, ly);
        let p3 = (rx - 8.0, ry);

        let shared_origin = wire_origin_ports.iter().position(|&p| p == port_idx);
        let shared_dest =
            dest_indices.get(..i).and_then(|slice| slice.iter().position(|&d| d == dest_idx));

        let mut best_ctrl = [p0, p0, p3, p3];
        let mut best_coarse = [CoarsePoint::default(); COARSE_STEPS + 1];
        let mut best_penalty = f32::INFINITY;

        for _ in 0..MAX_COARSE_ATTEMPTS {
            let p1 = (
                PADDING + rng.random_range(68.0..165.0),
                PADDING + rng.random_range(0.0..(HEIGHT as f32 - 2.0 * PADDING)),
            );
            let p2 = (
                WIDTH as f32 - PADDING - rng.random_range(68.0..165.0),
                PADDING + rng.random_range(0.0..(HEIGHT as f32 - 2.0 * PADDING)),
            );

            let coarse = eval_coarse_bezier(p0, p1, p2, p3);
            let (ok, penalty) =
                check_coarse_clarity(&coarse, &accepted_coarse, shared_origin, shared_dest);

            if penalty < best_penalty {
                best_penalty = penalty;
                best_ctrl = [p0, p1, p2, p3];
                best_coarse = coarse;
            }
            if ok && penalty < 20.0 {
                break;
            }
        }

        let [p0, p1, p2, p3] = best_ctrl;
        accepted_coarse.push(best_coarse);
        clear_beziers.push(best_ctrl);
        wire_origin_ports.push(port_idx);
        all_wires.push(sample_bezier_by_arc_length(p0, p1, p2, p3, 1.0, &mut rng));
        dash_phases.push(rng.random_range(0.0..34.0));
        wire_modulations.push(StrokeModulation::random(&mut rng));
    }

    // 3. Sample unconnected "dead-end" wires originating from valid or decoy left ports that terminate mid-canvas
    for dead_idx in 0..dead_end_wires {
        // Originate either from an unused decoy left port or bifurcate from a valid left port
        let port_idx = if dead_idx == 0 && total_left_ports > full_span_wires {
            total_left_ports - 1
        } else {
            rng.random_range(0..real_wires)
        };
        let &(lx, ly) = left_port_coords.get(port_idx).unwrap_or(&(14.0, HEIGHT as f32 * 0.5));
        let p0 = (lx + 18.0, ly);
        let shared_origin = wire_origin_ports.iter().position(|&p| p == port_idx);

        let mut best_ctrl = [p0, p0, p0, p0];
        let mut best_coarse = [CoarsePoint::default(); COARSE_STEPS + 1];
        let mut best_penalty = f32::INFINITY;

        for _ in 0..MAX_COARSE_ATTEMPTS {
            let end_x = WIDTH as f32 * rng.random_range(0.42..0.76);
            let end_y = rng.random_range((PADDING + 12.0)..(HEIGHT as f32 - PADDING - 12.0));
            let p3 = (end_x, end_y);
            let span_x = (end_x - p0.0).max(60.0);

            let p1 = (
                p0.0 + span_x * rng.random_range(0.28..0.45),
                PADDING + rng.random_range(0.0..(HEIGHT as f32 - 2.0 * PADDING)),
            );
            let p2 = (
                p0.0 + span_x * rng.random_range(0.62..0.82),
                PADDING + rng.random_range(0.0..(HEIGHT as f32 - 2.0 * PADDING)),
            );

            let coarse = eval_coarse_bezier(p0, p1, p2, p3);
            let (ok, penalty) =
                check_coarse_clarity(&coarse, &accepted_coarse, shared_origin, None);

            if penalty < best_penalty {
                best_penalty = penalty;
                best_ctrl = [p0, p1, p2, p3];
                best_coarse = coarse;
            }
            if ok && penalty < 25.0 {
                break;
            }
        }

        let [p0, p1, p2, p3] = best_ctrl;
        accepted_coarse.push(best_coarse);
        clear_beziers.push(best_ctrl);
        wire_origin_ports.push(port_idx);
        all_wires.push(sample_bezier_by_arc_length(p0, p1, p2, p3, 1.0, &mut rng));
        dash_phases.push(rng.random_range(0.0..34.0));
        wire_modulations.push(StrokeModulation::random(&mut rng));
    }

    apply_underpass_swerves(&mut all_wires, UNDERPASS_CUT_RADIUS);

    // Place sequential badges along each full-span wire + extra decoy badges around random arcs
    let (per_wire_badges, mut all_badges) = select_wire_badges(
        &all_wires,
        full_span_wires,
        &query_indices,
        badges_per_wire,
        waypoint_pool,
        &mut pool_cursor,
        &mut rng,
    );
    let decoy_arc_badges = select_decoy_arc_badges(
        &all_wires,
        &mut all_badges,
        extra_decoy_badges,
        WIDTH,
        HEIGHT,
        PADDING,
        waypoint_pool,
        &mut pool_cursor,
        &mut rng,
    );

    // Build the correct answer and dynamic prompt matching the active configuration
    let mut correct_answer = String::with_capacity(queried_count * (badges_per_wire + 1));
    for &w_idx in &query_indices {
        if let Some(badges) = per_wire_badges.get(w_idx) {
            for badge in badges {
                correct_answer.push_str(badge.digit);
            }
        }
        if let Some(&dest_idx) = dest_indices.get(w_idx)
            && let Some(&label) = right_labels.get(dest_idx)
        {
            correct_answer.push_str(label);
        }
    }

    let wires_list_str = if let &[q0, q1] = query_indices.as_slice() {
        let l0 = left_labels.get(q0).copied().unwrap_or("1");
        let l1 = left_labels.get(q1).copied().unwrap_or("2");
        format!("{} and {}", l0, l1)
    } else {
        let mut s = String::new();
        for (pos, &w_idx) in query_indices.iter().enumerate() {
            if pos > 0 {
                if pos + 1 == query_indices.len() {
                    s.push_str(" and ");
                } else {
                    s.push_str(", ");
                }
            }
            if let Some(&label) = left_labels.get(w_idx) {
                s.push_str(label);
            }
        }
        s
    };

    let expected_chars = queried_count * (badges_per_wire + 1);
    let sample_code = match (queried_count, badges_per_wire) {
        (2, 1) => "7B4e",
        (2, 2) => "73B49e",
        (2, 3) => "735B492e",
        (3, 1) => "7B4e2C",
        (3, 2) => "73B49e26C",
        _ => "735B492e864C",
    };
    let prompt = if badges_per_wire == 1 {
        format!(
            "Trace wires {}: enter each wire's waypoint character followed by its right-end character ({} chars, e.g. {}, base32 chars {:?}, the answer can be in either upper- or lowercase)",
            wires_list_str,
            expected_chars,
            sample_code,
            BASE32_ALPHABET.join("")
        )
    } else {
        format!(
            "Trace wires {}: enter each wire's {} characters from left to right followed by its right-end character ({} chars, e.g. {}, base32 chars {:?}, the answer can be in either upper- or lowercase)",
            wires_list_str,
            badges_per_wire,
            expected_chars,
            sample_code,
            BASE32_ALPHABET.join("")
        )
    };

    let pt_clear_of_badges_and_wires = |x: f32, y: f32| {
        if all_badges.iter().any(|b| dist_sq(x, y, b.x as f32, b.y as f32) <= BADGE_HALO_RADIUS_SQ)
        {
            return false;
        }
        // Stop micro-strokes at UNDERPASS_CUT_RADIUS from real wires so they look like genuine
        // underpassing wire fragments without muddying the foreground trajectories.
        !all_wires.iter().any(|wire| {
            wire.iter().step_by(3).any(|wpt| {
                (x - wpt.x).abs() < UNDERPASS_CUT_RADIUS
                    && (y - wpt.y).abs() < UNDERPASS_CUT_RADIUS
                    && dist_sq(x, y, wpt.x, wpt.y) < UNDERPASS_CUT_RADIUS_SQ
            })
        })
    };

    let grid_min_x = PADDING + 10.0;
    let grid_max_x = (WIDTH as f32) - PADDING - 10.0;
    let grid_min_y = 14.0_f32;
    let grid_max_y = (HEIGHT as f32) - 14.0;

    // 1. Draw curved decoy arcs passing directly through each decoy badge's halo so
    // decoy numbers have realistic stroke entry/exit tangents and modulated stroke weights.
    for decoy_badge in &decoy_arc_badges {
        let bx = decoy_badge.x as f32;
        let by = decoy_badge.y as f32;
        let theta0 = decoy_badge.ty.atan2(decoy_badge.tx);
        let curvature = rng.random_range(-0.025_f32..0.025_f32);
        let half_len: i32 = rng.random_range(18..=26);
        let modulation = StrokeModulation::random(&mut rng);

        let mut prev_pt: Option<(f32, f32)> = None;
        for step in -half_len..=half_len {
            let s = step as f32;
            let ang = theta0 + curvature * s;
            let x = bx + s * ang.cos();
            let y = by + s * ang.sin();

            if !(grid_min_x..=grid_max_x).contains(&x) || !(grid_min_y..=grid_max_y).contains(&y) {
                prev_pt = None;
                continue;
            }

            let d2_self = dist_sq(x, y, bx, by);
            let visible = d2_self > BADGE_HALO_RADIUS_SQ
                && (d2_self <= BADGE_SOLID_LOCK_RADIUS_SQ || step.rem_euclid(19) < 12)
                && pt_clear_of_badges_and_wires(x, y);

            if visible {
                if let Some((px, py)) = prev_pt {
                    let nx = -ang.sin();
                    let ny = ang.cos();
                    let radius = modulation.radius_at((step + half_len) as f32);
                    draw_modulated_segment_mut(
                        &mut img,
                        (px, py),
                        (x, y),
                        nx,
                        ny,
                        radius,
                        stroke_color,
                    );
                }
                prev_pt = Some((x, y));
            } else {
                prev_pt = None;
            }
        }
    }

    // 2. Draw multi-scale curved micro-strokes across the background using stratified grid cells
    // with dynamic stroke weight modulation.
    if bg_micro_stroke_count > 0 {
        bg_micro_stroke_count += rng.random_range(0..=10);
        let span_x = (grid_max_x - grid_min_x).max(1.0);
        let span_y = (grid_max_y - grid_min_y).max(1.0);
        let aspect = span_x / span_y;

        // Compute stratified grid dimensions (cols x rows >= bg_micro_stroke_count)
        let cols = ((bg_micro_stroke_count as f32 * aspect).sqrt().ceil() as usize).max(1);
        let rows = bg_micro_stroke_count.div_ceil(cols).max(1);
        let cell_w = span_x / (cols as f32);
        let cell_h = span_y / (rows as f32);

        let scale = BG_MICRO_STROKE_LEN_SCALE.clamp(0.25, 4.0);

        for idx in 0..bg_micro_stroke_count {
            // Map idx to a stratified cell so all regions of the screen receive strokes
            let cell_idx = (idx * cols * rows) / bg_micro_stroke_count;
            let col = cell_idx % cols;
            let row = cell_idx / cols;
            let cell_x0 = grid_min_x + (col as f32) * cell_w;
            let cell_y0 = grid_min_y + (row as f32) * cell_h;

            // Try up to 6 candidates inside this cell to start in an open pocket clear of wires/badges
            let mut x = cell_x0 + 0.5 * cell_w;
            let mut y = cell_y0 + 0.5 * cell_h;
            for _ in 0..6 {
                let cand_x = cell_x0 + rng.random_range(0.08..0.92) * cell_w;
                let cand_y = cell_y0 + rng.random_range(0.08..0.92) * cell_h;
                x = cand_x;
                y = cand_y;
                if pt_clear_of_badges_and_wires(cand_x, cand_y) {
                    break;
                }
            }

            let mut theta = rng.random_range(-PI..PI);
            let curvature = rng.random_range(-0.065_f32..0.065_f32);
            let wave_freq = rng.random_range(0.18_f32..0.32_f32);
            let wave_phase = rng.random_range(0.0..2.0 * PI);
            let modulation = StrokeModulation::random(&mut rng);

            let (steps, gap_range) = match idx % 3 {
                0 => {
                    let s = ((rng.random_range(6..=10) as f32) * scale).round() as i32;
                    (s.max(3), -1..-1)
                }
                1 => {
                    let s = ((rng.random_range(12..=18) as f32) * scale).round() as i32;
                    (s.max(5), -1..-1)
                }
                _ => {
                    let total = ((rng.random_range(22..=30) as f32) * scale).round() as i32;
                    let total = total.max(10);
                    let gap_len = (5.0 * scale).round().clamp(3.0, (total / 3) as f32) as i32;
                    let gap_start =
                        rng.random_range((total / 4)..=(total - gap_len - 2).max(total / 4));
                    (total, gap_start..(gap_start + gap_len))
                }
            };

            let mut prev_pt: Option<(f32, f32)> = None;
            for step in 0..steps {
                // Reflect smoothly off canvas bounds instead of breaking early
                if x < grid_min_x || x > grid_max_x {
                    x = x.clamp(grid_min_x, grid_max_x);
                    theta = PI - theta;
                    prev_pt = None;
                }
                if y < grid_min_y || y > grid_max_y {
                    y = y.clamp(grid_min_y, grid_max_y);
                    theta = -theta;
                    prev_pt = None;
                }

                let visible = !gap_range.contains(&step) && pt_clear_of_badges_and_wires(x, y);
                if visible {
                    if let Some((px, py)) = prev_pt {
                        let nx = -theta.sin();
                        let ny = theta.cos();
                        let radius = modulation.radius_at(step as f32);
                        draw_modulated_segment_mut(
                            &mut img,
                            (px, py),
                            (x, y),
                            nx,
                            ny,
                            radius,
                            stroke_color,
                        );
                    }
                    prev_pt = Some((x, y));
                } else {
                    prev_pt = None;
                }

                theta += curvature + 0.04 * (step as f32 * wave_freq + wave_phase).sin();
                x += theta.cos();
                y += theta.sin();
            }
        }
    }

    // 3. Render intertwined wires with over/under bridge cuts, empty number halos, and dynamic stroke weights
    for (i, ((wire, &dash_phase), modulation)) in
        all_wires.iter().zip(dash_phases.iter()).zip(wire_modulations.iter()).enumerate()
    {
        let mut prev_visible: Option<SamplePoint> = None;

        for pt in wire.iter() {
            // Cut a clean empty halo around any badge; lock wire dashes solid right outside its own badges
            let mut in_badge_halo = false;
            let mut near_own_badge = false;
            for b in &all_badges {
                let d2 = dist_sq(pt.x, pt.y, b.x as f32, b.y as f32);
                if d2 <= BADGE_HALO_RADIUS_SQ {
                    in_badge_halo = true;
                    break;
                }
                if b.wire_idx == Some(i) && d2 <= BADGE_SOLID_LOCK_RADIUS_SQ {
                    near_own_badge = true;
                }
            }

            if in_badge_halo {
                prev_visible = None;
                continue;
            }

            let mut is_underpass_cut = false;
            let mut is_overpass_lock = near_own_badge;

            for (j, other_wire) in all_wires.iter().enumerate() {
                if i == j {
                    continue;
                }
                // Step by 2 along 1px-sampled other_wire (max error 0.5px, cuts inner loop work in half)
                for other_pt in other_wire.iter().step_by(2) {
                    if (pt.x - other_pt.x).abs() >= OVERPASS_SOLID_RADIUS
                        || (pt.y - other_pt.y).abs() >= OVERPASS_SOLID_RADIUS
                    {
                        continue;
                    }
                    let d2 = dist_sq(pt.x, pt.y, other_pt.x, other_pt.y);
                    if d2 < OVERPASS_SOLID_RADIUS_SQ {
                        let mid_x = 0.5 * (pt.x + other_pt.x);
                        let z_self = wire_elevation(i, mid_x);
                        let z_other = wire_elevation(j, mid_x);

                        if z_self < z_other && d2 < UNDERPASS_CUT_RADIUS_SQ {
                            is_underpass_cut = true;
                            break;
                        } else if z_self >= z_other {
                            is_overpass_lock = true;
                        }
                    }
                }
                if is_underpass_cut {
                    break;
                }
            }

            let visible = if is_underpass_cut {
                false
            } else if is_overpass_lock {
                true
            } else {
                is_dash_on(pt.arc_len, dash_phase)
            };

            if visible {
                if let Some(prev) = prev_visible {
                    let radius = modulation.radius_at(pt.arc_len);
                    draw_modulated_segment_mut(
                        &mut img,
                        (prev.x, prev.y),
                        (pt.x, pt.y),
                        pt.nx,
                        pt.ny,
                        radius,
                        stroke_color,
                    );
                }
                prev_visible = Some(*pt);
            } else {
                prev_visible = None;
            }
        }
    }

    let font_data = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/minimal.ttf"));
    let font = FontRef::try_from_slice(font_data).map_err(|_| {
        zenwebp::EncodeError::IoError(std::io::Error::other("Error loading embedded font"))
    })?;
    let port_px_scale = 20.0_f32;
    let badge_px_scale = 19.5_f32;

    // Render mid-wire and decoy numbers with randomized contrast polarity (inverted vs normal),
    // affine shear (+/- 15 deg), and gentle rotation (+/- 10 deg).
    for badge in &all_badges {
        let inverted_contrast = rng.random_bool(0.5);
        let (halo_fill, glyph_fg) =
            if inverted_contrast { (stroke_color, bg_color) } else { (bg_color, stroke_color) };
        draw_filled_circle_mut(&mut img, (badge.x, badge.y), 10, halo_fill);

        let tangent_angle = badge.ty.atan2(badge.tx);
        draw_distorted_glyph_mut(
            &mut img,
            &font,
            badge.digit,
            badge.x,
            badge.y,
            badge_px_scale,
            tangent_angle,
            glyph_fg,
            &mut rng,
        );
    }

    // Draw randomized left port labels (real ports 1..=real_wires and decoy ports real_wires+1..=total_left_ports)
    for (&(lx, ly), &label) in left_port_coords.iter().zip(left_labels.iter()) {
        let cx = (lx + 5.0).round() as i32;
        let cy = ly.round() as i32;
        if rng.random_bool(0.45) {
            draw_filled_circle_mut(&mut img, (cx, cy), 10, stroke_color);
            draw_distorted_glyph_mut(
                &mut img,
                &font,
                label,
                cx,
                cy,
                port_px_scale,
                0.0,
                bg_color,
                &mut rng,
            );
        } else {
            draw_distorted_glyph_mut(
                &mut img,
                &font,
                label,
                cx,
                cy,
                port_px_scale,
                0.0,
                stroke_color,
                &mut rng,
            );
        }
    }

    // Draw randomized right port labels (randomly selected from BASE32_ALPHABET)
    for (&(rx, ry), &label) in right_port_coords.iter().zip(right_labels.iter()) {
        let cx = (rx + 5.0).round() as i32;
        let cy = ry.round() as i32;
        if rng.random_bool(0.45) {
            draw_filled_circle_mut(&mut img, (cx, cy), 10, stroke_color);
            draw_distorted_glyph_mut(
                &mut img,
                &font,
                label,
                cx,
                cy,
                port_px_scale,
                0.0,
                bg_color,
                &mut rng,
            );
        } else {
            draw_distorted_glyph_mut(
                &mut img,
                &font,
                label,
                cx,
                cy,
                port_px_scale,
                0.0,
                stroke_color,
                &mut rng,
            );
        }
    }

    let webp_bytes = EncodeRequest::lossless(
        // "cargo bench image_generation/generate_path_captcha" is now 5 ms
        // instead of 15 ms with defaults
        &LosslessConfig::new().with_quality(0.0).with_method(0),
        &img,
        PixelLayout::Rgb8,
        WIDTH,
        HEIGHT,
    )
    .encode()?;

    let challenge = CaptchaChallenge {
        webp_bytes,
        prompt: prompt.clone(),
        correct_answer: correct_answer.clone(),
    };

    let debug_challenge = if debug_overlay {
        draw_debug_overlay(&mut img, &clear_beziers, &per_wire_badges, &query_indices);
        let debug_webp_bytes =
            EncodeRequest::lossless(&LosslessConfig::new(), &img, PixelLayout::Rgb8, WIDTH, HEIGHT)
                .encode()?;
        Some(CaptchaChallenge {
            webp_bytes: debug_webp_bytes,
            prompt: prompt.clone(),
            correct_answer: correct_answer.clone(),
        })
    } else {
        None
    };

    Ok((challenge, debug_challenge))
}
