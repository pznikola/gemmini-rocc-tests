// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "include/mxint8_golden.h"

#define TEST_M 3
#define TEST_N 5
#define TEST_K 65
#define TEST_K_BLOCKS ((TEST_K + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE)
#define IDENTITY_SCALE ((mx_scale_t)(127 + MX_INT_FRAC_BITS))

static acc_t saturate_expected(int64_t value) {
  if (value > INT32_MAX)
    return (acc_t)INT32_MAX;
  if (value < INT32_MIN)
    return (acc_t)INT32_MIN;
  return (acc_t)value;
}

int main(void) {
  if (mxint8_e8m0_decode((mx_scale_t)127) != 0)
    exit(1);
  if (mxint8_e8m0_decode(IDENTITY_SCALE) != MX_INT_FRAC_BITS)
    exit(1);
  if (mxint8_e8m0_is_valid((mx_scale_t)0xff))
    exit(1);

  if (mxint8_round_right_shift_nearest_even(1, 1) != 0)
    exit(1);
  if (mxint8_round_right_shift_nearest_even(3, 1) != 2)
    exit(1);
  if (mxint8_round_right_shift_nearest_even(-3, 1) != -2)
    exit(1);

  static elem_t a[TEST_M][TEST_K];
  static elem_t b[TEST_K][TEST_N];
  static mx_scale_t a_scale[TEST_M][TEST_K_BLOCKS];
  static mx_scale_t b_scale[TEST_K_BLOCKS][TEST_N];
  static acc_t c[TEST_M][TEST_N];

  for (size_t i = 0; i < TEST_M; i++)
    for (size_t kk = 0; kk < TEST_K; kk++)
      a[i][kk] = (elem_t)(((int)(i * 7 + kk * 3) % 17) - 8);

  for (size_t kk = 0; kk < TEST_K; kk++)
    for (size_t j = 0; j < TEST_N; j++)
      b[kk][j] = (elem_t)(((int)(kk * 5 + j * 11) % 19) - 9);

  for (size_t i = 0; i < TEST_M; i++)
    for (size_t block = 0; block < TEST_K_BLOCKS; block++)
      a_scale[i][block] = IDENTITY_SCALE;

  for (size_t block = 0; block < TEST_K_BLOCKS; block++)
    for (size_t j = 0; j < TEST_N; j++)
      b_scale[block][j] = IDENTITY_SCALE;

  if (mxint8_ref_gemm_acc(&a[0][0], &b[0][0], &a_scale[0][0], &b_scale[0][0], &c[0][0],
                          TEST_M, TEST_N, TEST_K,
                          TEST_K, TEST_N, TEST_N,
                          TEST_K_BLOCKS, TEST_N) != 0)
    exit(1);

  for (size_t i = 0; i < TEST_M; i++) {
    for (size_t j = 0; j < TEST_N; j++) {
      int64_t expected = 0;
      for (size_t kk = 0; kk < TEST_K; kk++)
        expected += (int32_t)a[i][kk] * (int32_t)b[kk][j];
      if (c[i][j] != saturate_expected(expected))
        exit(1);
    }
  }

  for (size_t i = 0; i < TEST_M; i++)
    for (size_t block = 0; block < TEST_K_BLOCKS; block++)
      a_scale[i][block] = (mx_scale_t)(IDENTITY_SCALE + ((i + block) & 1));

  for (size_t block = 0; block < TEST_K_BLOCKS; block++)
    for (size_t j = 0; j < TEST_N; j++)
      b_scale[block][j] = (mx_scale_t)(IDENTITY_SCALE - ((j + block) & 1));

  if (mxint8_ref_gemm_acc(&a[0][0], &b[0][0], &a_scale[0][0], &b_scale[0][0], &c[0][0],
                          TEST_M, TEST_N, TEST_K,
                          TEST_K, TEST_N, TEST_N,
                          TEST_K_BLOCKS, TEST_N) != 0)
    exit(1);

  for (size_t i = 0; i < TEST_M; i++) {
    for (size_t j = 0; j < TEST_N; j++) {
      int64_t expected = 0;
      for (size_t block = 0; block < TEST_K_BLOCKS; block++) {
        int64_t raw = 0;
        for (size_t t = 0; t < MX_BLOCK_SIZE; t++) {
          const size_t kk = block * MX_BLOCK_SIZE + t;
          if (kk < TEST_K)
            raw += (int32_t)a[i][kk] * (int32_t)b[kk][j];
        }
        expected += mxint8_scale_raw_block(raw,
            mxint8_e8m0_decode(a_scale[i][block]),
            mxint8_e8m0_decode(b_scale[block][j]));
      }
      if (c[i][j] != saturate_expected(expected))
        exit(1);
    }
  }

  a_scale[1][0] = (mx_scale_t)0xff;
  if (mxint8_ref_gemm_acc(&a[0][0], &b[0][0], &a_scale[0][0], &b_scale[0][0], &c[0][0],
                          TEST_M, TEST_N, TEST_K,
                          TEST_K, TEST_N, TEST_N,
                          TEST_K_BLOCKS, TEST_N) == 0)
    exit(1);

  printf("mxint8 golden tests passed\n");
  exit(0);
}
