use std::{env, error::Error, sync::Arc};

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    routing::{get, post},
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use bezcaptcha::{
    AppState, CHALLENGE_TTL_SECS, ChallengeResponse, DEFAULT_MAX_SPENT_NONCES, VerifyRequest,
    VerifyResponse,
    captcha::{Hardness, generate_path_captcha},
    create_captcha_token, current_unix_time, verify_captcha_token,
};
use rand::Rng;
use serde::Deserialize;

#[derive(Deserialize, Default)]
struct ChallengeQuery {
    hardness: Option<Hardness>,
}

// GET /api/captcha/challenge (supports optional ?hardness=easy|medium|hard)
async fn get_challenge_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ChallengeQuery>,
) -> Result<Json<ChallengeResponse>, StatusCode> {
    let hardness = query.hardness.unwrap_or(state.hardness);

    // Run CPU-intensive curve sampling and WebP encoding on Tokio's blocking threadpool
    let challenge = tokio::task::spawn_blocking(move || generate_path_captcha(hardness))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let now = current_unix_time();
    let expires_at = now + CHALLENGE_TTL_SECS;
    let token =
        create_captcha_token(&challenge.correct_answer, expires_at, &state.mac_secret, hardness);

    let base64_webp = BASE64.encode(&challenge.webp_bytes);
    let image_data_url = format!("data:image/webp;base64,{}", base64_webp);

    Ok(Json(ChallengeResponse {
        prompt: challenge.prompt,
        image_data_url,
        token,
        expires_in_secs: CHALLENGE_TTL_SECS,
    }))
}

// POST /api/captcha/verify
async fn verify_challenge_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyRequest>,
) -> (StatusCode, Json<VerifyResponse>) {
    let now = current_unix_time();

    match verify_captcha_token(&req.answer, &req.token, &state, now) {
        Ok(()) => (StatusCode::OK, Json(VerifyResponse { valid: true, reason: "Verified" })),
        Err(reason) => (StatusCode::BAD_REQUEST, Json(VerifyResponse { valid: false, reason })),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Generate a fresh 256-bit secret key on startup (or load from an environment variable)
    let mut secret = [0u8; 32];
    rand::rng().fill_bytes(&mut secret);

    // Configure default hardness from CLI arg or CAPTCHA_HARDNESS env var (defaults to Hard)
    let hardness = match env::args().nth(1).or_else(|| env::var("CAPTCHA_HARDNESS").ok()) {
        Some(val) => val.parse::<Hardness>()?,
        None => Hardness::default(),
    };

    let state = AppState::with_config(
        secret,
        DEFAULT_MAX_SPENT_NONCES,
        hardness,
        5, // Maximum failed attempts per token
    );

    let app = Router::new()
        .route("/api/captcha/challenge", get(get_challenge_handler))
        .route("/api/captcha/verify", post(verify_challenge_handler))
        .with_state(state.into());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;

    println!("CAPTCHA server listening on http://127.0.0.1:3000 (default hardness: {})", hardness);
    axum::serve(listener, app).await?;
    Ok(())
}
