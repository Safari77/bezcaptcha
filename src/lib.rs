//! # bezcaptcha
//!
//! Dual-licensed under AGPL-3.0-only or commercial terms.
//! See CONTRIBUTING.md for contributor licensing details.

pub mod captcha;

use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

pub use captcha::Hardness;
use rand::Rng;
use serde::{Deserialize, Serialize};
use shake::{
    Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};
use subtle::ConstantTimeEq;

/// Maximum lifetime of an issued CAPTCHA challenge in seconds (2 minutes).
///
/// Tokens whose expiration timestamp exceeds `now + CHALLENGE_TTL_SECS`
/// or whose expiration timestamp has already passed will be rejected during verification.
pub const CHALLENGE_TTL_SECS: u64 = 120;

/// Output length in bytes of the SHAKE256 message authentication code (MAC).
pub const MAC_OUTPUT_LEN: usize = 32;

/// Default maximum number of spent challenge nonces tracked across all ring-buffer buckets.
///
/// Limits peak memory consumption for spent nonce tracking to approximately 5 MB.
pub const DEFAULT_MAX_SPENT_NONCES: usize = 200_000;

const BUCKET_COUNT: usize = 3;
const BUCKET_DURATION_SECS: u64 = 60;

/// Version prefix prepended as the initial byte in the MAC computation payload.
///
/// Allows breaking format migrations or algorithm upgrades while invalidating older tokens.
pub const CAPTCHA_MAC_VERSION: u8 = 1;

/// Default maximum number of consecutive failed verification attempts allowed per challenge nonce
/// before rejecting further attempts.
pub const DEFAULT_MAX_FAILED_ATTEMPTS: u32 = 5;

/// Minimum cooldown interval in seconds required between verification attempts for the same token nonce.
pub const ATTEMPT_INTERVAL_SECS: u64 = 3;

/// Maximum allowed length in bytes of an answer string before trimming or validation.
pub const MAX_ANSWER_BYTES: usize = 64;

const NONCE_HEX_LEN: usize = 32;
const SIGNATURE_HEX_LEN: usize = MAC_OUTPUT_LEN * 2;
const MAX_TIMESTAMP_LEN: usize = 20; // u64::MAX has 20 decimal digits

/// Maximum permissible byte length of a formatted token string.
///
/// Layout: `nonce (32) + ':' + timestamp (up to 20) + ':' + issue_mac (64) + ':' + answer_mac (64)`.
pub const MAX_TOKEN_LEN: usize =
    NONCE_HEX_LEN + 1 + MAX_TIMESTAMP_LEN + 1 + SIGNATURE_HEX_LEN + 1 + SIGNATURE_HEX_LEN;

const MAX_ISSUE_PAYLOAD_LEN: usize = NONCE_HEX_LEN + 1 + MAX_TIMESTAMP_LEN;
const MAX_PAYLOAD_LEN: usize = NONCE_HEX_LEN + 1 + MAX_TIMESTAMP_LEN + 1 + MAX_ANSWER_BYTES;

const ATTEMPT_PRUNE_INTERVAL_SECS: u64 = 30;

struct AttemptState {
    failed_attempts: u32,
    last_attempt_unix: u64,
    expires_at_unix: u64,
}

// Bounded and periodically pruned, so attempt tracking cannot grow indefinitely.
struct AttemptStore {
    entries: HashMap<[u8; 16], AttemptState>,
    max_entries: usize,
    max_failed_attempts: u32,
    next_prune_unix: u64,
}

impl AttemptStore {
    fn new(max_entries: usize, max_failed_attempts: u32) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            max_failed_attempts: max_failed_attempts.max(1),
            next_prune_unix: 0,
        }
    }

    fn prune_if_needed(&mut self, now_unix: u64) {
        if now_unix < self.next_prune_unix {
            return;
        }

        self.entries.retain(|_, attempt| attempt.expires_at_unix >= now_unix);

        self.next_prune_unix = now_unix.saturating_add(ATTEMPT_PRUNE_INTERVAL_SECS);
    }

    // Reserves an attempt before any expensive verification work.
    fn begin_attempt(
        &mut self,
        nonce: [u8; 16],
        expires_at_unix: u64,
        now_unix: u64,
    ) -> Result<(), &'static str> {
        self.prune_if_needed(now_unix);

        if let Some(attempt) = self.entries.get_mut(&nonce) {
            if attempt.failed_attempts >= self.max_failed_attempts {
                return Err("Too many failed attempts");
            }

            let next_allowed = attempt.last_attempt_unix.saturating_add(ATTEMPT_INTERVAL_SECS);

            if now_unix < next_allowed {
                return Err("Attempts are limited to once every 3 seconds");
            }

            attempt.last_attempt_unix = now_unix;
            attempt.expires_at_unix = attempt.expires_at_unix.max(expires_at_unix);

            return Ok(());
        }

        if self.entries.len() >= self.max_entries {
            return Err("Attempt store capacity exceeded");
        }

        self.entries.insert(
            nonce,
            AttemptState { failed_attempts: 0, last_attempt_unix: now_unix, expires_at_unix },
        );

        Ok(())
    }

    fn record_failure(&mut self, nonce: &[u8; 16]) {
        if let Some(attempt) = self.entries.get_mut(nonce) {
            attempt.failed_attempts = attempt.failed_attempts.saturating_add(1);
        }
    }

    fn record_success(&mut self, nonce: &[u8; 16]) {
        self.entries.remove(nonce);
    }
}

/// Bounded generational ring-buffer to store spent nonces with O(1) expiry and zero heap keys
pub struct SpentNonceStore {
    current_epoch: u64,
    buckets: [HashSet<[u8; 16]>; BUCKET_COUNT],
    max_nonces: usize,
}

impl SpentNonceStore {
    pub fn new(max_nonces: usize) -> Self {
        Self { current_epoch: 0, buckets: Default::default(), max_nonces }
    }

    pub fn total_count(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    fn rotate_if_needed(&mut self, now_unix: u64) {
        let now_epoch = now_unix / BUCKET_DURATION_SECS;
        if self.current_epoch == 0 {
            self.current_epoch = now_epoch;
            return;
        }
        if now_epoch <= self.current_epoch {
            return;
        }

        let elapsed = (now_epoch - self.current_epoch) as usize;
        if elapsed >= BUCKET_COUNT {
            for bucket in &mut self.buckets {
                bucket.clear();
            }
        } else {
            for _ in 0..elapsed {
                self.buckets.rotate_right(1);
                self.buckets[0].clear();
            }
        }
        self.current_epoch = now_epoch;
    }

    pub fn check_and_insert(&mut self, nonce: [u8; 16], now_unix: u64) -> Result<(), &'static str> {
        self.rotate_if_needed(now_unix);

        for bucket in &self.buckets {
            if bucket.contains(&nonce) {
                return Err("Token already used");
            }
        }

        if self.total_count() >= self.max_nonces {
            return Err("Nonce store capacity exceeded");
        }

        self.buckets[0].insert(nonce);
        Ok(())
    }
}

/// Shared server application state managing secrets, challenge configuration, and anti-replay stores.
///
/// Tracks spent nonces and rate-limited verification attempts behind thread-safe locks.
pub struct AppState {
    pub mac_secret: [u8; 32],
    pub hardness: Hardness,
    pub spent_nonces: Mutex<SpentNonceStore>,
    attempts: Mutex<AttemptStore>,
}

impl AppState {
    /// Constructs a new `AppState` with a custom spent nonce capacity and hardness level.
    ///
    /// Sets the maximum failed attempts to [`DEFAULT_MAX_FAILED_ATTEMPTS`].
    ///
    /// # Arguments
    ///
    /// * `mac_secret` - 32-byte secret key for signing tokens.
    /// * `max_spent_nonces` - Maximum number of nonces tracked before eviction/rejection.
    /// * `hardness` - Target difficulty level for CAPTCHA challenges.
    pub fn with_capacity_and_hardness(
        mac_secret: [u8; 32],
        max_spent_nonces: usize,
        hardness: Hardness,
    ) -> Self {
        Self::with_config(mac_secret, max_spent_nonces, hardness, DEFAULT_MAX_FAILED_ATTEMPTS)
    }

    /// Constructs a new `AppState` with fully custom parameters.
    ///
    /// # Arguments
    ///
    /// * `mac_secret` - 32-byte secret key for signing tokens.
    /// * `max_spent_nonces` - Maximum number of nonces tracked before eviction/rejection.
    /// * `hardness` - Target difficulty level for CAPTCHA challenges.
    /// * `max_failed_attempts` - Maximum allowable failed attempts before a challenge is locked.
    pub fn with_config(
        mac_secret: [u8; 32],
        max_spent_nonces: usize,
        hardness: Hardness,
        max_failed_attempts: u32,
    ) -> Self {
        Self {
            mac_secret,
            hardness,
            spent_nonces: Mutex::new(SpentNonceStore::new(max_spent_nonces)),
            attempts: Mutex::new(AttemptStore::new(max_spent_nonces, max_failed_attempts)),
        }
    }
}

/// Outgoing challenge payload sent to the client.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChallengeResponse {
    /// Human-readable prompt or instructions for the CAPTCHA challenge.
    pub prompt: String,
    /// Data URL encoding the visual CAPTCHA image (e.g., `data:image/png;base64,...`).
    pub image_data_url: String,
    /// Signed state token containing the nonce, expiration, and authentication MACs.
    pub token: String,
    /// Duration in seconds for which this challenge remains valid.
    pub expires_in_secs: u64,
}

/// Incoming client request payload containing the solution attempt.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VerifyRequest {
    /// Signed state token originally issued with the challenge.
    pub token: String,
    /// Plaintext answer submitted by the user.
    pub answer: String,
}

/// Result payload returned following a verification attempt.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VerifyResponse {
    /// Indicates whether the challenge was successfully solved.
    pub valid: bool,
    /// Status description or error explanation (e.g., `"ok"`, `"Incorrect answer"`, `"Challenge expired"`).
    pub reason: &'static str,
}

/// Seconds since UNIX epoch
pub fn current_unix_time() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Computes a fixed-size SHAKE256 message authentication code (MAC).
///
/// Prepends [`CAPTCHA_MAC_VERSION`], length-prefixed domain tags, length-prefixed secret bytes,
/// and an explicit hardness tag before absorbing the payload to prevent domain confusion and length extension.
///
/// # Arguments
///
/// * `secret` - The symmetric secret key.
/// * `hardness` - Challenge hardness setting encoded as a static domain tag.
/// * `domain` - Context-separating domain slice (e.g., `b"issue"` or `b"answer"`).
/// * `payload` - Message bytes to authenticate.
///
/// # Returns
///
/// A 32-byte array containing the evaluated MAC digest.
pub fn compute_shake256_mac(
    secret: &[u8],
    hardness: Hardness,
    domain: &[u8],
    payload: &[u8],
) -> [u8; MAC_OUTPUT_LEN] {
    // Explicit tags avoid depending on enum discriminant or display ordering.
    let hardness_tag: u8 = match hardness {
        Hardness::Easy => 0,
        Hardness::Medium => 1,
        Hardness::Hard => 2,
    };

    let mut hasher = Shake256::default();

    // Version is deliberately the first byte of the MAC input.
    hasher.update(&[CAPTCHA_MAC_VERSION]);
    hasher.update(b"captcha");
    // Domain separation distinguishes issuance tags from answer tags
    hasher.update(&(domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update(&(secret.len() as u64).to_be_bytes());
    hasher.update(secret);
    hasher.update(&[hardness_tag]);
    hasher.update(payload);

    let mut reader = hasher.finalize_xof();
    let mut output = [0u8; MAC_OUTPUT_LEN];
    reader.read(&mut output);
    output
}

/// Generates a signed CAPTCHA token containing challenge state and verification tags.
///
/// Generates a cryptographically random 16-byte nonce, normalizes the answer (trimmed and uppercased),
/// and produces a token with format:
/// `<nonce_hex>:<expires_at_unix>:<issue_mac_hex>:<answer_mac_hex>`.
///
/// # Arguments
///
/// * `answer` - The expected plaintext solution.
/// * `expires_at_unix` - Unix timestamp in seconds when the token becomes invalid.
/// * `secret` - Symmetric signing key.
/// * `hardness` - Challenge hardness level bound into the MAC.
///
/// # Returns
///
/// Colon-delimited hex token string.
pub fn create_captcha_token(
    answer: &str,
    expires_at_unix: u64,
    secret: &[u8],
    hardness: Hardness,
) -> String {
    let mut nonce_bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut nonce_bytes);

    let nonce = hex::encode(nonce_bytes);
    let normalized_answer = answer.trim().to_ascii_uppercase();

    // 1. MAC proving this challenge was genuinely issued by the server
    let issue_payload = format!("{}:{}", nonce, expires_at_unix);
    let issue_mac = compute_shake256_mac(secret, hardness, b"issue", issue_payload.as_bytes());

    // 2. MAC binding the valid answer to the nonce and expiry
    let answer_payload = format!("{}:{}:{}", nonce, expires_at_unix, normalized_answer);
    let answer_mac = compute_shake256_mac(secret, hardness, b"answer", answer_payload.as_bytes());

    format!("{}:{}:{}:{}", nonce, expires_at_unix, hex::encode(issue_mac), hex::encode(answer_mac))
}

/// Verifies a submitted CAPTCHA token and user answer against a specified hardness level.
///
/// Evaluates validity through the following defense pipeline:
/// 1. Bounds-checks raw input lengths against [`MAX_TOKEN_LEN`] and [`MAX_ANSWER_BYTES`].
/// 2. Validates timestamp freshness within `[now, now + CHALLENGE_TTL_SECS]`.
/// 3. Validates the issuance MAC before modifying state to prevent forged-token rate-limit exhaustion.
/// 4. Checks attempt cooldown ([`ATTEMPT_INTERVAL_SECS`]) and limits via `AttemptStore`.
/// 5. Validates the answer MAC against normalized user input in constant time.
/// 6. Marks the nonce as spent in [`SpentNonceStore`] to protect against replay attacks.
///
/// # Arguments
///
/// * `user_input` - The raw answer provided by the user.
/// * `token` - The serialized token string.
/// * `state` - Current server [`AppState`].
/// * `hardness` - Target difficulty level to verify against.
/// * `now_unix` - Current Unix timestamp in seconds.
///
/// # Errors
///
/// Returns `Err(&'static str)` if the token is malformed, expired, forged, rate-limited,
/// already spent, or if the answer is incorrect.
pub fn verify_captcha_token_with_hardness(
    user_input: &str,
    token: &str,
    state: &AppState,
    hardness: Hardness,
    now_unix: u64,
) -> Result<(), &'static str> {
    // Check raw input lengths before trim(), parsing, or normalization.
    if token.len() > MAX_TOKEN_LEN {
        return Err("Malformed token");
    }
    if user_input.len() > MAX_ANSWER_BYTES {
        return Err("Answer too long");
    }

    // Iterator parsing: no Vec and no input-proportional allocation.
    let mut parts = token.split(':');

    let nonce = parts.next().ok_or("Malformed token")?;
    let expires_at_str = parts.next().ok_or("Malformed token")?;
    let issue_mac_hex = parts.next().ok_or("Malformed token")?;
    let answer_mac_hex = parts.next().ok_or("Malformed token")?;

    if parts.next().is_some()
        || nonce.len() != NONCE_HEX_LEN
        || expires_at_str.is_empty()
        || expires_at_str.len() > MAX_TIMESTAMP_LEN
        || issue_mac_hex.len() != SIGNATURE_HEX_LEN
        || answer_mac_hex.len() != SIGNATURE_HEX_LEN
    {
        return Err("Malformed token");
    }

    let expires_at: u64 = expires_at_str.parse().map_err(|_| "Invalid timestamp")?;

    if now_unix > expires_at {
        return Err("Challenge expired");
    }

    if expires_at > now_unix.saturating_add(CHALLENGE_TTL_SECS) {
        return Err("Invalid timestamp");
    }

    // Decode directly into fixed-size stack arrays.
    let mut nonce_bytes = [0u8; 16];
    hex::decode_to_slice(nonce, &mut nonce_bytes).map_err(|_| "Malformed token")?;

    let mut issue_signature_bytes = [0u8; MAC_OUTPUT_LEN];
    hex::decode_to_slice(issue_mac_hex, &mut issue_signature_bytes)
        .map_err(|_| "Invalid signature encoding")?;

    // Construct nonce:expires_at on the stack to verify issuance authenticity before touching state.
    let mut issue_payload = [0u8; MAX_ISSUE_PAYLOAD_LEN];
    let mut issue_payload_len = 0usize;

    for field in [nonce.as_bytes(), b":", expires_at_str.as_bytes()] {
        let end = issue_payload_len + field.len();
        let dest = issue_payload.get_mut(issue_payload_len..end).ok_or("Malformed token")?;
        dest.copy_from_slice(field);
        issue_payload_len = end;
    }

    let issue_payload_slice = issue_payload.get(..issue_payload_len).ok_or("Malformed token")?;
    let expected_issue_mac =
        compute_shake256_mac(&state.mac_secret, hardness, b"issue", issue_payload_slice);

    // Reject forged tokens here. No locks are held and AttemptStore is completely untouched.
    if expected_issue_mac.ct_eq(&issue_signature_bytes).unwrap_u8() != 1 {
        return Err("Malformed or forged token");
    }

    let mut answer_signature_bytes = [0u8; MAC_OUTPUT_LEN];
    hex::decode_to_slice(answer_mac_hex, &mut answer_signature_bytes)
        .map_err(|_| "Invalid signature encoding")?;

    // Reserve attempt only after authentic issuance is proven, preventing forged-token bucket exhaustion.
    {
        let mut attempts = state.attempts.lock().map_err(|_| "Attempt store unavailable")?;

        attempts.begin_attempt(nonce_bytes, expires_at, now_unix)?;
    }

    let answer = user_input.trim();
    let answer_len = answer.len(); // Guaranteed <= MAX_ANSWER_BYTES above.

    let mut normalized_answer = [0u8; MAX_ANSWER_BYTES];
    let norm_slice = normalized_answer.get_mut(..answer_len).ok_or("Answer too long")?;
    norm_slice.copy_from_slice(answer.as_bytes());
    norm_slice.make_ascii_uppercase();

    let mut payload = [0u8; MAX_PAYLOAD_LEN];
    let mut payload_len = 0usize;

    let fields: [&[u8]; 5] = [nonce.as_bytes(), b":", expires_at_str.as_bytes(), b":", norm_slice];

    for field in fields {
        let end = payload_len + field.len();
        let dest = payload.get_mut(payload_len..end).ok_or("Malformed token")?;
        dest.copy_from_slice(field);
        payload_len = end;
    }

    let payload_slice = payload.get(..payload_len).ok_or("Malformed token")?;
    let expected_answer_mac =
        compute_shake256_mac(&state.mac_secret, hardness, b"answer", payload_slice);

    if expected_answer_mac.ct_eq(&answer_signature_bytes).unwrap_u8() != 1 {
        let mut attempts = state.attempts.lock().map_err(|_| "Attempt store unavailable")?;

        attempts.record_failure(&nonce_bytes);

        return Err("Incorrect answer");
    }

    // The nonce is marked spent before the successful result is returned.
    {
        let mut spent = state.spent_nonces.lock().map_err(|_| "Nonce store unavailable")?;

        spent.check_and_insert(nonce_bytes, now_unix)?;
    }

    let mut attempts = state.attempts.lock().map_err(|_| "Attempt store unavailable")?;

    attempts.record_success(&nonce_bytes);

    Ok(())
}

/// Verifies a submitted CAPTCHA token using the default hardness level configured in [`AppState`].
///
/// Convenience wrapper around [`verify_captcha_token_with_hardness`].
///
/// # Arguments
///
/// * `user_input` - The raw answer provided by the user.
/// * `token` - The serialized token string.
/// * `state` - Current server [`AppState`] whose `hardness` field is used.
/// * `now_unix` - Current Unix timestamp in seconds.
///
/// # Errors
///
/// Returns `Err(&'static str)` on verification failure, expiration, rate-limiting, or invalid token structure.
pub fn verify_captcha_token(
    user_input: &str,
    token: &str,
    state: &AppState,
    now_unix: u64,
) -> Result<(), &'static str> {
    verify_captcha_token_with_hardness(user_input, token, state, state.hardness, now_unix)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_forged_issuance_mac_does_not_leak_into_attempt_store() {
        let secret = [0x11u8; 32];
        let state = AppState::with_capacity_and_hardness(secret, 10, Hardness::Easy);
        let now = 1_000_000;
        let expires_at = now + 60;

        // Craft a forged token with arbitrary nonces and bogus MACs
        let bogus_issue_mac = "a".repeat(SIGNATURE_HEX_LEN);
        let bogus_answer_mac = "b".repeat(SIGNATURE_HEX_LEN);
        let forged_token = format!(
            "{:032x}:{}:{}:{}",
            0xdeadbeef_u64, expires_at, bogus_issue_mac, bogus_answer_mac
        );

        let res =
            verify_captcha_token_with_hardness("ANS1", &forged_token, &state, Hardness::Easy, now);
        assert_eq!(res, Err("Malformed or forged token"));

        // Verify AttemptStore contains zero entries and was never locked/inserted into
        let attempts = state.attempts.lock().unwrap();
        assert_eq!(attempts.entries.len(), 0);
    }

    #[test]
    fn test_forged_tokens_cannot_exhaust_attempt_store_capacity() {
        let secret = [0x22u8; 32];
        // Constrain capacity to 3 entries to test DoS resilience
        let state = AppState::with_capacity_and_hardness(secret, 3, Hardness::Easy);
        let now = 1_000_000;
        let expires_at = now + 60;

        // Flood 20 distinct forged tokens, far exceeding the capacity limit of 3
        for i in 0..20 {
            let forged_token =
                format!("{:032x}:{}:{:064x}:{:064x}", i, expires_at, 0x1234_u64, 0x5678_u64);
            let res = verify_captcha_token_with_hardness(
                "ANS",
                &forged_token,
                &state,
                Hardness::Easy,
                now,
            );
            // Must fail issuance validation rather than triggering "Attempt store capacity exceeded"
            assert_eq!(res, Err("Malformed or forged token"));
        }

        // Verify AttemptStore remained completely unpolluted
        let attempts = state.attempts.lock().unwrap();
        assert_eq!(attempts.entries.len(), 0);
    }

    #[test]
    fn test_valid_token_wrong_answer_records_attempt() {
        let secret = [0x33u8; 32];
        let state = AppState::with_capacity_and_hardness(secret, 10, Hardness::Easy);
        let now = 1_000_000;
        let expires_at = now + 60;

        let token = create_captcha_token("REAL_ANSWER", expires_at, &secret, Hardness::Easy);

        let res =
            verify_captcha_token_with_hardness("WRONG_ANSWER", &token, &state, Hardness::Easy, now);
        assert_eq!(res, Err("Incorrect answer"));

        // Authentic challenge issuance passes, so the failed attempt is recorded
        let attempts = state.attempts.lock().unwrap();
        assert_eq!(attempts.entries.len(), 1);
        let attempt_state = attempts.entries.values().next().unwrap();
        assert_eq!(attempt_state.failed_attempts, 1);
    }

    #[test]
    fn test_valid_token_success_flow_clears_attempt_and_spends_nonce() {
        let secret = [0x44u8; 32];
        let state = AppState::with_capacity_and_hardness(secret, 10, Hardness::Easy);
        let now = 1_000_000;
        let expires_at = now + 60;

        let token = create_captcha_token("CORRECT", expires_at, &secret, Hardness::Easy);

        let res =
            verify_captcha_token_with_hardness("correct", &token, &state, Hardness::Easy, now);
        assert_eq!(res, Ok(()));

        // Attempt entry is removed upon success
        let attempts = state.attempts.lock().unwrap();
        assert_eq!(attempts.entries.len(), 0);

        // Nonce is tracked as spent in SpentNonceStore
        let spent = state.spent_nonces.lock().unwrap();
        assert_eq!(spent.total_count(), 1);
    }
}
