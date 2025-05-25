#![allow(unstable_features)]
#![allow(unsafe_op_in_unsafe_fn)]
#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]
use criterion::{criterion_group, criterion_main, Criterion, black_box};
use core::simd::Simd;
use std::simd::cmp::SimdPartialOrd;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Scalar fallback
fn relu_scalar(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
}

/// Portable SIMD
fn relu_simd(input: &[f32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());
    let chunks = input.chunks_exact(16);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let v = Simd::<f32, 16>::from_slice(chunk);
        let mask = v.simd_gt(Simd::splat(0.0));
        let res = mask.select(v, Simd::splat(0.0));
        output.extend(res.to_array());
    }

    for &x in remainder {
        output.push(if x > 0.0 { x } else { 0.0 });
    }

    output
}

/// AVX-512 SIMD
#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn relu_avx512(input: &[f32]) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(input.len());
    let chunks: std::slice::ChunksExact<'_, f32> = input.chunks_exact(16);
    let remainder: &[f32] = chunks.remainder();

    for chunk in chunks {
        let v: __m512 = _mm512_loadu_ps(chunk.as_ptr());
        let zero: __m512 = _mm512_set1_ps(0.0);
        let mask: u16 = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
        let result: __m512 = _mm512_mask_blend_ps(mask, zero, v);

        let mut tmp = [0f32; 16];
        _mm512_storeu_ps(tmp.as_mut_ptr(), result);
        output.extend_from_slice(&tmp);
    }

    for &x in remainder {
        output.push(if x > 0.0 { x } else { 0.0 });
    }

    output
}

/// Criterion benchmarks
fn benchmark_relu(c: &mut Criterion) {
    let input: Vec<f32> = (0..1_000_000).map(|i| (i as f32 % 20.0) - 10.0).collect();

    c.bench_function("scalar", |b| b.iter(|| relu_scalar(black_box(&input))));
    c.bench_function("portable SIMD", |b| b.iter(|| relu_simd(black_box(&input))));

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            c.bench_function("AVX-512", |b| {
                b.iter(|| unsafe { relu_avx512(black_box(&input)) });
            });
        }
        else 
        {
            println!("AVX-512 not supported on this architecture.");
        }
    }
}

criterion_group!(benches, benchmark_relu);
criterion_main!(benches);