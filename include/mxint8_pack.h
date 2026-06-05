// See LICENSE for license details.
//
// MXINT8 packer: quantize dense fp32 matrices into the block-scaled MXINT8
// representation consumed by both the software golden (`mxint8_golden.h`) and
// the Gemmini hardware scale-mvin path (`gemmini_mvin_mxscale_a/b`).
//
// For a 32-element MX block the packer picks the smallest E8M0 power-of-two
// shared scale `2^(e-6)` that keeps every signed-int8 payload in the symmetric
// range `[-127, 127]`, then rounds each element to nearest-even. A row of A is
// quantized per K block (`A_scale[M][ceil(K/32)]`); a column of B is quantized
// per K block (`B_scale[ceil(K/32)][N]`), matching `mxint8_policy.md`.

#ifndef GEMMINI_MXINT8_PACK_H
#define GEMMINI_MXINT8_PACK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "include/gemmini_params.h"
#include "include/mxint8_golden.h"

/**
 * Build `2^m` as an exact float via IEEE-754 exponent construction (no libm).
 *
 * @param m Integer exponent; clamped to the normalized single-precision range.
 * @return The value `2^m`.
 */
static inline float mxint8_exp2f_int(int m) {
  if (m > 127) {
    m = 127;
  }
  if (m < -126) {
    return 0.0f;
  }

  union {
    float f;
    uint32_t u;
  } v;
  v.u = (uint32_t)(m + 127) << 23; // 1.0 * 2^m
  return v.f;
}

/**
 * Smallest integer `p` such that `2^p >= x`, for `x > 0` (no libm).
 *
 * @param x A strictly positive, normalized float.
 * @return `ceil(log2(x))`.
 */
static inline int mxint8_ceil_log2f(float x) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = x;

  const int floor_exp = (int)((v.u >> 23) & 0xffu) - 127; // floor(log2(x))
  const uint32_t mantissa = v.u & 0x7fffffu;
  return (mantissa == 0u) ? floor_exp : floor_exp + 1; // exact power of two?
}

/**
 * Encode an integer exponent as an E8M0 scale byte (inverse of
 * `mxint8_e8m0_decode`). The exponent is clamped to `[-127, 127]` so the result
 * is never the reserved NaN byte `0xff`.
 *
 * @param exp Decoded exponent `e`.
 * @return E8M0 scale byte `e + 127`, guaranteed valid.
 */
static inline mx_scale_t mxint8_e8m0_encode(int exp) {
  if (exp > 127) {
    exp = 127;
  }
  if (exp < -127) {
    exp = -127;
  }
  return (mx_scale_t)(exp + 127);
}

/**
 * Quantize a single MX block of up to `MX_BLOCK_SIZE` fp32 values into signed
 * int8 payloads sharing one E8M0 scale.
 *
 * @param vals Pointer to `n` contiguous fp32 inputs.
 * @param n Number of valid elements (`1 <= n <= MX_BLOCK_SIZE`).
 * @param payload Output buffer for `n` signed-int8 payloads.
 * @return The shared E8M0 scale byte for the block.
 */
static inline mx_scale_t mxint8_quantize_block(const float *vals, size_t n,
                                               elem_t *payload) {
  float amax = 0.0f;
  for (size_t t = 0; t < n; t++) {
    const float a = vals[t] < 0.0f ? -vals[t] : vals[t];
    if (a > amax) {
      amax = a;
    }
  }

  // Smallest exponent `e` with payload step `2^(e-6) >= amax/127`, so the block
  // maximum maps to <= 127. An all-zero block takes the neutral scale `e = 0`.
  const int exp = (amax == 0.0f) ? 0 : mxint8_ceil_log2f(amax / 127.0f) + MX_INT_FRAC_BITS;
  const float inv_step = mxint8_exp2f_int(MX_INT_FRAC_BITS - exp); // 2^(6-e)

  for (size_t t = 0; t < n; t++) {
    int32_t q = (int32_t)ROUND_NEAR_EVEN(vals[t] * inv_step);
    if (q > 127) {
      q = 127;
    }
    if (q < -127) {
      q = -127;
    }
    payload[t] = (elem_t)q;
  }

  return mxint8_e8m0_encode(exp);
}

/**
 * Pack a dense row-major fp32 matrix `A[M][K]` into MXINT8 payloads and
 * per-row, per-block A scales.
 *
 * Payload layout matches the input (`a_payload[i*a_stride + k]`); scales are
 * `a_scale[i*a_scale_stride + block]` — one scale row per M row, blocks across
 * the lanes, exactly as `gemmini_mvin_mxscale_a` loads them.
 *
 * @param a Dense fp32 input, row stride `a_stride`.
 * @param m Number of rows.
 * @param k Number of columns (K).
 * @param a_stride Element stride between A rows (payload uses the same stride).
 * @param a_scale_stride Element stride between A-scale rows (>= ceil(k/32)).
 * @param a_payload Output signed-int8 payloads, row stride `a_stride`.
 * @param a_scale Output E8M0 scales, row stride `a_scale_stride`.
 */
static inline void mxint8_pack_a(const float *a, size_t m, size_t k,
                                 size_t a_stride, size_t a_scale_stride,
                                 elem_t *a_payload, mx_scale_t *a_scale) {
  const size_t k_blocks = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;

  for (size_t i = 0; i < m; i++) {
    for (size_t b = 0; b < k_blocks; b++) {
      const size_t base = b * MX_BLOCK_SIZE;
      const size_t n = (base + MX_BLOCK_SIZE <= k) ? MX_BLOCK_SIZE : (k - base);
      // A block is contiguous in K, so quantize in place.
      a_scale[i * a_scale_stride + b] =
          mxint8_quantize_block(&a[i * a_stride + base], n, &a_payload[i * a_stride + base]);
    }
  }
}

/**
 * Pack a dense row-major fp32 matrix `B[K][N]` into MXINT8 payloads and
 * per-column, per-block B scales.
 *
 * Each MX block runs down the K dimension within one column, so the block
 * elements are strided by `b_stride`; they are gathered, quantized, and
 * scattered back. Scales are `b_scale[block*b_scale_stride + j]` — one scale row
 * per K block, columns across the lanes, exactly as `gemmini_mvin_mxscale_b`
 * loads them.
 *
 * @param b Dense fp32 input, row stride `b_stride`.
 * @param k Number of rows (K).
 * @param n Number of columns (N).
 * @param b_stride Element stride between B rows (payload uses the same stride).
 * @param b_scale_stride Element stride between B-scale rows (>= n).
 * @param b_payload Output signed-int8 payloads, row stride `b_stride`.
 * @param b_scale Output E8M0 scales, row stride `b_scale_stride`.
 */
static inline void mxint8_pack_b(const float *b, size_t k, size_t n,
                                 size_t b_stride, size_t b_scale_stride,
                                 elem_t *b_payload, mx_scale_t *b_scale) {
  const size_t k_blocks = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;

  for (size_t j = 0; j < n; j++) {
    for (size_t blk = 0; blk < k_blocks; blk++) {
      const size_t base = blk * MX_BLOCK_SIZE;
      const size_t rows = (base + MX_BLOCK_SIZE <= k) ? MX_BLOCK_SIZE : (k - base);

      float vals[MX_BLOCK_SIZE];
      elem_t payload[MX_BLOCK_SIZE];
      for (size_t t = 0; t < rows; t++) {
        vals[t] = b[(base + t) * b_stride + j];
      }

      b_scale[blk * b_scale_stride + j] = mxint8_quantize_block(vals, rows, payload);

      for (size_t t = 0; t < rows; t++) {
        b_payload[(base + t) * b_stride + j] = payload[t];
      }
    }
  }
}

#endif // GEMMINI_MXINT8_PACK_H
