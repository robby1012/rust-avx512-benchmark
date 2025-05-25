#![allow(unstable_features)]
#![allow(unsafe_op_in_unsafe_fn)]
#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]
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

fn main() {
    let input: Vec<f32> = (0..1_000_000).map(|i| (i as f32 % 20.0) - 10.0).collect();
    
    // Warm up
    let _ = relu_scalar(&input);
    let _ = relu_simd(&input);
    unsafe { let _ = relu_avx512(&input); }

    // Benchmark
    use std::time::Instant;

    println!("Benchmarking ReLU implementations with {} elements:", input.len());

    let start = Instant::now();
    let scalar_result = relu_scalar(&input);
    let scalar_duration = start.elapsed();
    println!("Scalar implementation: {:?}", scalar_duration);

    let start = Instant::now();
    let simd_result = relu_simd(&input);
    let simd_duration = start.elapsed();
    println!("Portable SIMD implementation: {:?}", simd_duration);

    // Verify results match
    assert_eq!(scalar_result, simd_result);
    if is_x86_feature_detected!("avx512f") {
        let start = Instant::now();
        let avx512_result = unsafe { relu_avx512(&input) };
        let avx512_duration = start.elapsed();
        println!("AVX-512 implementation: {:?}", avx512_duration);
        assert_eq!(scalar_result, avx512_result);
    } else {
        println!("AVX-512 not supported on this architecture.");
    }
    println!("All implementations produced identical results!");
}
