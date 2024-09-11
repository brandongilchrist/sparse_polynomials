use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ark_ff::UniformRand;
use ark_test_curves::bls12_381::Fr;
use ark_std::rand::Rng;
use sparse_poly::{SparsePoly, evaluate_multiple, evaluate_multiple_parallel, evaluate_multiple_hybrid, set_parallel_threshold};

fn generate_random_poly_data<R: Rng>(rng: &mut R, size: usize, sparsity: f64) -> Vec<(Fr, u64)> {
    (0..size)
        .filter_map(|i| {
            if rng.gen::<f64>() < sparsity {
                Some((Fr::rand(rng), i as u64))
            } else {
                None
            }
        })
        .collect()
}

fn create_random_poly<R: Rng>(rng: &mut R, degree: u64, sparsity: f64) -> SparsePoly {
    let pairs = generate_random_poly_data(rng, degree as usize + 1, sparsity);
    SparsePoly::new(pairs)
}

fn benchmark_evaluate(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let poly_small = create_random_poly(&mut rng, 1 << 20, 0.01);
    let poly_large = create_random_poly(&mut rng, 1 << 24, 0.001);
    let x = Fr::rand(&mut rng);

    c.bench_function("evaluate small poly", |b| {
        b.iter(|| black_box(&poly_small).evaluate(black_box(&x)))
    });

    c.bench_function("evaluate large poly", |b| {
        b.iter(|| black_box(&poly_large).evaluate(black_box(&x)))
    });
}

fn benchmark_parallel_evaluate(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let poly_small = create_random_poly(&mut rng, 1 << 20, 0.01);
    let poly_large = create_random_poly(&mut rng, 1 << 24, 0.001);
    let x = Fr::rand(&mut rng);

    c.bench_function("parallel evaluate small poly", |b| {
        b.iter(|| black_box(&poly_small).evaluate_parallel(black_box(&x)))
    });

    c.bench_function("parallel evaluate large poly", |b| {
        b.iter(|| black_box(&poly_large).evaluate_parallel(black_box(&x)))
    });
}

fn benchmark_cached_evaluate(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let mut poly_small = create_random_poly(&mut rng, 1 << 20, 0.01);
    let mut poly_large = create_random_poly(&mut rng, 1 << 24, 0.001);
    let x = Fr::rand(&mut rng);

    poly_small.precompute_powers(&x);
    poly_large.precompute_powers(&x);

    c.bench_function("cached evaluate small poly", |b| {
        b.iter(|| black_box(&poly_small).evaluate(black_box(&x)))
    });

    c.bench_function("cached evaluate large poly", |b| {
        b.iter(|| black_box(&poly_large).evaluate(black_box(&x)))
    });
}

fn benchmark_hybrid_evaluate(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let poly_small = create_random_poly(&mut rng, 1 << 20, 0.01);
    let poly_large = create_random_poly(&mut rng, 1 << 24, 0.001);
    let x = Fr::rand(&mut rng);

    c.bench_function("hybrid evaluate small poly", |b| {
        b.iter(|| black_box(&poly_small).evaluate_hybrid(black_box(&x)))
    });

    c.bench_function("hybrid evaluate large poly", |b| {
        b.iter(|| black_box(&poly_large).evaluate_hybrid(black_box(&x)))
    });

    let mut cached_poly_large = poly_large.clone();
    cached_poly_large.precompute_powers(&x);
    c.bench_function("hybrid evaluate large poly with cache", |b| {
        b.iter(|| black_box(&cached_poly_large).evaluate_hybrid(black_box(&x)))
    });
}

fn benchmark_evaluate_multiple(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let polys: Vec<SparsePoly> = (0..100)
        .map(|_| create_random_poly(&mut rng, 1 << 20, 0.01))
        .collect();
    let x = Fr::rand(&mut rng);

    c.bench_function("evaluate multiple polys", |b| {
        b.iter(|| evaluate_multiple(black_box(&polys), black_box(&x)))
    });

    c.bench_function("parallel evaluate multiple polys", |b| {
        b.iter(|| evaluate_multiple_parallel(black_box(&polys), black_box(&x)))
    });

    c.bench_function("hybrid evaluate multiple polys", |b| {
        b.iter(|| evaluate_multiple_hybrid(black_box(&polys), black_box(&x)))
    });
}

fn benchmark_parallel_threshold(c: &mut Criterion) {
    let thresholds = [100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000];
    let mut rng = ark_std::test_rng();
    let poly_large = create_random_poly(&mut rng, 1 << 24, 0.001);
    let x = Fr::rand(&mut rng);

    let mut group = c.benchmark_group("parallel_threshold");
    for threshold in thresholds.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(threshold), threshold, |b, &threshold| {
            set_parallel_threshold(threshold);
            b.iter(|| black_box(&poly_large).evaluate_hybrid(black_box(&x)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_evaluate,
    benchmark_parallel_evaluate,
    benchmark_cached_evaluate,
    benchmark_hybrid_evaluate,
    benchmark_evaluate_multiple,
    benchmark_parallel_threshold
);
criterion_main!(benches);