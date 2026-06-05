// See LICENSE for license details.

#ifndef GEMMINI_MXINT8_GOLDEN_H
#define GEMMINI_MXINT8_GOLDEN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>

#include "include/gemmini_params.h"

static inline bool mxint8_e8m0_is_valid(mx_scale_t scale) {
  return scale != (mx_scale_t)0xff;
}

static inline mx_exp_t mxint8_e8m0_decode(mx_scale_t scale) {
  return (mx_exp_t)((int)scale - 127);
}

static inline int64_t mxint8_round_right_shift_nearest_even(int64_t value, unsigned shift) {
  if (shift == 0)
    return value;

  if (shift >= 63)
    return 0;

  const bool negative = value < 0;
  const uint64_t magnitude = negative ? (uint64_t)(-value) : (uint64_t)value;
  const uint64_t quotient = magnitude >> shift;
  const uint64_t remainder = magnitude & (((uint64_t)1 << shift) - 1);
  const uint64_t halfway = (uint64_t)1 << (shift - 1);
  uint64_t rounded = quotient;

  if (remainder > halfway || (remainder == halfway && (quotient & 1)))
    rounded++;

  return negative ? -(int64_t)rounded : (int64_t)rounded;
}

static inline int64_t mxint8_scale_raw_block(int64_t raw, mx_exp_t exp_a, mx_exp_t exp_b) {
  const int shift = (int)exp_a + (int)exp_b - (2 * MX_INT_FRAC_BITS);

  if (shift >= 0) {
    if (shift >= 62)
      return raw < 0 ? INT64_MIN : INT64_MAX;
    if (raw > (INT64_MAX >> shift))
      return INT64_MAX;
    if (raw < (INT64_MIN >> shift))
      return INT64_MIN;
    return raw << shift;
  }

  return mxint8_round_right_shift_nearest_even(raw, (unsigned)(-shift));
}

static inline acc_t mxint8_saturate_acc(int64_t value) {
  if (value > INT32_MAX)
    return (acc_t)INT32_MAX;
  if (value < INT32_MIN)
    return (acc_t)INT32_MIN;
  return (acc_t)value;
}

static inline int mxint8_ref_gemm_acc(const elem_t *a_payload,
                                      const elem_t *b_payload,
                                      const mx_scale_t *a_scale,
                                      const mx_scale_t *b_scale,
                                      acc_t *c_acc,
                                      size_t m, size_t n, size_t k,
                                      size_t a_stride, size_t b_stride, size_t c_stride,
                                      size_t a_scale_stride, size_t b_scale_stride) {
  const size_t k_blocks = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      int64_t acc = 0;

      for (size_t block = 0; block < k_blocks; block++) {
        const mx_scale_t scale_a = a_scale[i * a_scale_stride + block];
        const mx_scale_t scale_b = b_scale[block * b_scale_stride + j];

        if (!mxint8_e8m0_is_valid(scale_a) || !mxint8_e8m0_is_valid(scale_b))
          return -1;

        int64_t raw = 0;
        for (size_t t = 0; t < MX_BLOCK_SIZE; t++) {
          const size_t kk = block * MX_BLOCK_SIZE + t;
          if (kk < k)
            raw += (int32_t)a_payload[i * a_stride + kk] * (int32_t)b_payload[kk * b_stride + j];
        }

        acc += mxint8_scale_raw_block(raw, mxint8_e8m0_decode(scale_a), mxint8_e8m0_decode(scale_b));
      }

      c_acc[i * c_stride + j] = mxint8_saturate_acc(acc);
    }
  }

  return 0;
}

#endif
