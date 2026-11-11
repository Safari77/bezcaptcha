use std::{
    env,
    io::{self, Write},
    path::{Path, PathBuf},
    process,
};

use bezcaptcha::{
    AppState, CHALLENGE_TTL_SECS, DEFAULT_MAX_SPENT_NONCES,
    captcha::{Hardness, generate_path_captcha, generate_path_captcha_with_debug_overlay},
    create_captcha_token, current_unix_time, verify_captcha_token,
};
use rand::Rng;
use tempfile::NamedTempFile;

fn parse_hardness() -> Hardness {
    let mut args = env::args().skip(1);

    let hardness = match args.next() {
        Some(value) => value.parse::<Hardness>().unwrap_or_else(|err| {
            eprintln!("error: {err}");
            eprintln!("usage: captcha_generator [easy|medium|hard]");
            process::exit(2);
        }),
        None => Hardness::default(),
    };

    if let Some(unexpected) = args.next() {
        eprintln!("error: unexpected argument: {unexpected}");
        eprintln!("usage: captcha_generator [easy|medium|hard]");
        process::exit(2);
    }

    hardness
}

fn replace_file_atomically(path: &Path, contents: &[u8]) -> io::Result<()> {
    // Using the same directory ensures the rename stays on the same filesystem.
    let directory = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    let mut temporary = NamedTempFile::new_in(directory)?;
    temporary.write_all(contents)?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|err| err.error)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hardness = parse_hardness();
    let hardness_name = hardness.as_str();
    let unsolved_path = PathBuf::from(format!("test_captcha_{hardness_name}.webp"));
    let solved_path = PathBuf::from(format!("test_captcha_{hardness_name}_solved.webp"));

    println!("============================================================");
    println!("              Bézier CAPTCHA Test Generator & Checker       ");
    println!("============================================================");
    // 1. Generate challenge image
    println!("[1/3] Generating test CAPTCHA image...");
    println!("      Hardness:        {}", hardness);
    let (challenge, debug_challenge) = generate_path_captcha_with_debug_overlay(hardness)?;

    replace_file_atomically(&unsolved_path, &challenge.webp_bytes)
        .map_err(|err| format!("Failed to write {}: {}", unsolved_path.display(), err))?;
    replace_file_atomically(&solved_path, &debug_challenge.webp_bytes)
        .map_err(|err| format!("Failed to write {}: {}", solved_path.display(), err))?;
    println!(
        "      Saved WebP images to: {} ({} bytes) and {} ({} bytes)",
        unsolved_path.display(),
        challenge.webp_bytes.len(),
        solved_path.display(),
        debug_challenge.webp_bytes.len()
    );

    println!("      Prompt:          {}", challenge.prompt);
    println!("      Correct Answer:  {}", challenge.correct_answer);
    // 2. Initialize application state and mint a token
    println!("\n[2/3] Setting up test AppState and generating token...");
    let mut secret = [0u8; 32];
    rand::rng().fill_bytes(&mut secret);
    let state = AppState::with_config(
        secret,
        DEFAULT_MAX_SPENT_NONCES,
        hardness,
        5, // Maximum failed attempts per token
    );
    let now = current_unix_time();
    let expires_at = now + CHALLENGE_TTL_SECS;
    let token =
        create_captcha_token(&challenge.correct_answer, expires_at, &state.mac_secret, hardness);
    println!("      Minted Token: {}", token);
    // 3. Verify captcha check logic
    println!("\n[3/3] Running verification checks...");
    // Test A: Correct answer verification succeeds
    match verify_captcha_token(&challenge.correct_answer, &token, &state, now) {
        Ok(()) => println!("  [PASS] Correct answer accepted"),
        Err(err) => {
            return Err(
                format!("Expected verification to pass with correct answer, got: {}", err).into()
            );
        }
    }
    // Test B: Replay protection (single-use nonce) rejects reuse of the same token
    match verify_captcha_token(&challenge.correct_answer, &token, &state, now) {
        Err("Token already used") => println!("  [PASS] Single-use nonce enforced on token replay"),
        other => {
            return Err(format!("Expected 'Token already used', got: {:?}", other).into());
        }
    }
    // Test C: Incorrect answer verification fails
    let token_for_wrong_answer =
        create_captcha_token(&challenge.correct_answer, expires_at, &state.mac_secret, hardness);
    let wrong_answer = "0".repeat(challenge.correct_answer.len());
    match verify_captcha_token(&wrong_answer, &token_for_wrong_answer, &state, now) {
        Err("Incorrect answer") => {
            println!("  [PASS] Incorrect answer rejected");
        }
        other => {
            return Err(format!("Expected 'Incorrect answer', got: {:?}", other).into());
        }
    }
    // Test D: Case insensitivity and whitespace trimming
    let token_for_case_test =
        create_captcha_token(&challenge.correct_answer, expires_at, &state.mac_secret, hardness);
    let case_varied_answer = format!("  {}  ", challenge.correct_answer.to_ascii_lowercase());
    match verify_captcha_token(&case_varied_answer, &token_for_case_test, &state, now) {
        Ok(()) => println!("  [PASS] Case-insensitivity and whitespace trimming validated"),
        Err(err) => {
            return Err(format!("Expected lowercase/trimmed answer to pass, got: {}", err).into());
        }
    }
    // Test E: Expired challenge rejected
    let token_expired =
        create_captcha_token(&challenge.correct_answer, expires_at, &state.mac_secret, hardness);
    let past_expiry = expires_at + 1;
    match verify_captcha_token(&challenge.correct_answer, &token_expired, &state, past_expiry) {
        Err("Challenge expired") => println!("  [PASS] Expired token rejected"),
        other => {
            return Err(format!("Expected 'Challenge expired', got: {:?}", other).into());
        }
    }
    // Test F: Tampered token signature rejected
    let tampered_token = format!("{}deadbeef", token_expired);
    match verify_captcha_token(&challenge.correct_answer, &tampered_token, &state, now) {
        Err(_) => println!("  [PASS] Tampered token signature rejected"),
        Ok(()) => return Err("Expected tampered token to fail".into()),
    }
    // Test F2: Forged issuance MAC rejected
    let forged_token = format!(
        "{:032x}:{}:{:064x}:{:064x}",
        0xdeadbeef_u64, expires_at, 0x1111_2222_u64, 0x3333_4444_u64
    );
    match verify_captcha_token(&challenge.correct_answer, &forged_token, &state, now) {
        Err("Malformed or forged token") => {
            println!("  [PASS] Forged token rejected before attempt tracking");
        }
        other => {
            return Err(format!("Expected 'Malformed or forged token', got: {:?}", other).into());
        }
    }
    // Test G: Validate all three hardness levels and sorted wire numbers in prompts
    for level in [Hardness::Easy, Hardness::Medium, Hardness::Hard] {
        let ch = generate_path_captcha(level)?;
        if ch.webp_bytes.is_empty() {
            return Err(format!("Empty WebP for hardness {}", level).into());
        }

        let (queried_count, badges_per_wire, _) = level.query_config();
        let expected_len = queried_count * (badges_per_wire + 1);
        if ch.correct_answer.len() != expected_len {
            return Err(format!(
                "Answer should be {} chars for hardness {}, got {}",
                expected_len,
                level,
                ch.correct_answer.len()
            )
            .into());
        }

        // Verify wire numbers in "Trace wires X and Y: ..." are strictly ascending
        let after_prefix =
            ch.prompt.strip_prefix("Trace wires ").ok_or("Prompt missing 'Trace wires ' prefix")?;
        let wires_part = after_prefix.split(':').next().ok_or("Prompt missing ':'")?;
        let wire_nums: Vec<u32> = wires_part
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty() && *s != "and")
            .map(|s| s.parse::<u32>().map_err(|_| format!("Bad wire label '{}'", s)))
            .collect::<Result<Vec<u32>, _>>()?;
        if wire_nums.len() != queried_count {
            return Err(format!(
                "Expected {} queried wires in prompt, got {:?}",
                queried_count, wire_nums
            )
            .into());
        }
        if !wire_nums.is_sorted_by(|a, b| a < b) {
            return Err(
                format!("Expected sorted wire numbers in prompt, got {:?}", wire_nums).into()
            );
        }

        let lvl_state = AppState::with_config(state.mac_secret, DEFAULT_MAX_SPENT_NONCES, level, 5);
        let lvl_token =
            create_captcha_token(&ch.correct_answer, expires_at, &lvl_state.mac_secret, level);
        verify_captcha_token(&ch.correct_answer, &lvl_token, &lvl_state, now)
            .map_err(|e| format!("Verification failed for hardness {}: {}", level, e))?;
        println!(
            "  [PASS] Hardness level '{}' verified (prompt: \"Trace wires {}\")",
            level.as_str(),
            wires_part
        );
    }
    println!("\nAll CAPTCHA generation and verification checks passed successfully!");
    Ok(())
}
