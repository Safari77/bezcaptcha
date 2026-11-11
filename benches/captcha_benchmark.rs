#![allow(clippy::unwrap_used)]
use bezcaptcha::{
    AppState, Hardness, captcha::generate_path_captcha, create_captcha_token, verify_captcha_token,
};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

fn bench_image_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_generation");
    group.sample_size(10);

    group.bench_function("generate_path_captcha", |b| {
        b.iter(|| {
            std::hint::black_box(generate_path_captcha(Hardness::Medium).unwrap());
        });
    });

    group.finish();
}

fn bench_token_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_operations");
    let secret = [0x5au8; 32];
    let hardness = Hardness::default();
    let state = AppState::with_capacity_and_hardness(secret, 200_000, hardness);
    let now_unix = 1_700_000_000;
    let expires_at_unix = now_unix + 60;

    group.bench_function("create_captcha_token", |b| {
        b.iter(|| {
            std::hint::black_box(create_captcha_token(
                "73B49E",
                expires_at_unix,
                &secret,
                hardness,
            ));
        });
    });

    // Profile fast rejection of forged tokens (proves early rejection without mutex locks or heap allocation)
    let forged_token = format!(
        "{:032x}:{}:{:064x}:{:064x}",
        0xdeadbeef_u64, expires_at_unix, 0x1111_2222_u64, 0x3333_4444_u64
    );

    group.bench_function("verify_forged_token_rejection", |b| {
        b.iter(|| {
            let _ = std::hint::black_box(verify_captcha_token(
                "73B49E",
                &forged_token,
                &state,
                now_unix,
            ));
        });
    });

    // Profile valid token verification using iter_batched to provide a fresh, unspent token per iteration
    group.bench_function("verify_valid_token_success", |b| {
        b.iter_batched(
            || create_captcha_token("73B49E", expires_at_unix, &secret, hardness),
            |token| {
                let _ =
                    std::hint::black_box(verify_captcha_token("73B49E", &token, &state, now_unix));
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_image_generation, bench_token_operations);
criterion_main!(benches);
