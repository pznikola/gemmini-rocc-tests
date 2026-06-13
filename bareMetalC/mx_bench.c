// See LICENSE for license details.
//
// Stock-vs-MXINT8 benchmark (plan Stage C / P6 subset). ONE source, built per config via
// the header-staging pattern: with a stock params header (MX_ENABLED=0) the GEMM runs
// through `tiled_matmul_auto` (WS); with an MX header it runs through
// `tiled_matmul_mxint8`. Same shapes, same int8 payload data, same DIM per pair, so the
// MAC count is identical and the cycle delta is attributable to the MX metadata path.
//
// Fairness boundary: inputs are generated host-side at init (integer payloads + E8M0
// scales directly — quantization/packing is excluded on both sides; stock int8 needs
// none). The timed region is the tiled_matmul call + fence: payload mvins (+ scale
// mvins and per-invocation fences on the MX side), compute, and mvout. full_C=true on
// both sides (identical 4-byte/element mvout traffic).
//
// Correctness gates every data point: GOLDEN_ROWS sampled output rows are checked
// against the bit-exact reference (int32 GEMM for stock; `mxint8_ref_gemm_acc` rows for
// MX). Machine-readable result lines:
//   MXBENCH,impl=<stock|mx>,dim=<D>,M=,N=,K=,cycles=,macs=,ideal_cycles=,util_pct=,result=<PASS|FAIL>
//
// Shape table is gated by DIM to keep Verilator runtimes bounded (see plan Stage C).

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#if MX_ENABLED
#include "include/mxint8_golden.h"
#endif

#define GOLDEN_ROWS 8

// Shape limits (bytes are dominated by B and C for the BERT shapes).
#if DIM >= 8
#define M_MAX 512
#define N_MAX 3072
#define K_MAX 768
#else
#define M_MAX 256
#define N_MAX 256
#define K_MAX 256
#endif
#define KB_MAX ((K_MAX + 31) / 32)

typedef struct {
  size_t m, n, k;
} bench_shape_t;

static const bench_shape_t shapes[] = {
#if DIM >= 16
  {64, 64, 64},
  {128, 128, 128},
  {256, 256, 256},
  {512, 512, 512},
  {128, 768, 768},
  {128, 3072, 768},
#elif DIM == 8
  {64, 64, 64},
  {128, 128, 128},
  {256, 256, 256},
  {512, 512, 512},
  {128, 768, 768},
#else
  {64, 64, 64},
  {128, 128, 128},
  {256, 256, 256},
#endif
};

static elem_t a_payload[M_MAX][K_MAX] row_align(1);
static elem_t b_payload[K_MAX][N_MAX] row_align(1);
static acc_t c_hw[M_MAX][N_MAX] row_align(1);
static acc_t gold_row[N_MAX];
#if MX_ENABLED
static mx_scale_t a_scale[M_MAX][KB_MAX] __attribute__((aligned(64)));
static mx_scale_t b_scale[KB_MAX][N_MAX] __attribute__((aligned(64)));
#endif

static uint32_t lcg = 0xa5a5a5a5u;
static uint32_t lcg_next(void) {
  lcg = lcg * 1664525u + 1013904223u;
  return lcg;
}

// Integer payloads in the packer-reachable range [-127, 127] and small-exponent E8M0
// scales (so MX results stay far from the int32 saturation envelope).
static void gen_inputs(size_t m, size_t n, size_t k) {
  for (size_t i = 0; i < m; i++) {
    for (size_t kk = 0; kk < k; kk++) {
      a_payload[i][kk] = (elem_t)((int)(lcg_next() % 255) - 127);
    }
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < n; j++) {
      b_payload[kk][j] = (elem_t)((int)(lcg_next() % 255) - 127);
    }
  }
#if MX_ENABLED
  const size_t kb = (k + 31) / 32;
  for (size_t i = 0; i < m; i++) {
    for (size_t b = 0; b < kb; b++) {
      a_scale[i][b] = (mx_scale_t)(127 + (int)(lcg_next() % 5) - 2);  // exp in [-2, 2]
    }
  }
  for (size_t b = 0; b < kb; b++) {
    for (size_t j = 0; j < n; j++) {
      b_scale[b][j] = (mx_scale_t)(127 + (int)(lcg_next() % 5) - 2);
    }
  }
#endif
}

static void run_gemm(size_t m, size_t n, size_t k) {
#if MX_ENABLED
  tiled_matmul_mxint8(m, n, k,
      &a_payload[0][0], &b_payload[0][0], /*D=*/NULL, &c_hw[0][0],
      &a_scale[0][0], &b_scale[0][0],
      K_MAX, N_MAX, 0, N_MAX, KB_MAX, N_MAX,
      /*full_C=*/true, /*low_D=*/false, /*act=*/0);
#else
  tiled_matmul_auto(m, n, k,
      &a_payload[0][0], &b_payload[0][0], /*D=*/NULL, &c_hw[0][0],
      K_MAX, N_MAX, 0, N_MAX,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      /*act=*/NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, /*repeating_bias=*/false,
      /*transpose_A=*/false, /*transpose_B=*/false,
      /*full_C=*/true, /*low_D=*/false,
      /*weightA=*/3, WS);
#endif
  gemmini_fence();
}

// Bit-exact reference for one sampled output row.
static int check_row(size_t row, size_t n, size_t k) {
#if MX_ENABLED
  const int rc = mxint8_ref_gemm_acc(&a_payload[row][0], &b_payload[0][0],
                                     &a_scale[row][0], &b_scale[0][0], &gold_row[0],
                                     1, n, k, K_MAX, N_MAX, N_MAX, KB_MAX, N_MAX);
  if (rc != 0) {
    printf("golden rejected scale on row %lu (rc=%d)\n", (unsigned long)row, rc);
    return 1;
  }
#else
  for (size_t j = 0; j < n; j++) {
    acc_t sum = 0;
    for (size_t kk = 0; kk < k; kk++) {
      sum += (acc_t)a_payload[row][kk] * (acc_t)b_payload[kk][j];
    }
    gold_row[j] = sum;
  }
#endif
  for (size_t j = 0; j < n; j++) {
    if (c_hw[row][j] != gold_row[j]) {
      printf("row %lu col %lu: hw=%d gold=%d\n",
             (unsigned long)row, (unsigned long)j, (int)c_hw[row][j], (int)gold_row[j]);
      return 1;
    }
  }
  return 0;
}

static int run_shape(const bench_shape_t *s) {
  gen_inputs(s->m, s->n, s->k);

  gemmini_flush(0);
  const uint64_t start = read_cycles();
  run_gemm(s->m, s->n, s->k);
  const uint64_t end = read_cycles();

  int bad = 0;
  for (int r = 0; r < GOLDEN_ROWS; r++) {
    bad |= check_row(lcg_next() % s->m, s->n, s->k);
  }

  const uint64_t cycles = end - start;
  const uint64_t macs = (uint64_t)s->m * s->n * s->k;
  const uint64_t ideal = macs / ((uint64_t)DIM * DIM);
  const uint64_t util_pct = cycles ? (100 * ideal) / cycles : 0;

  printf("MXBENCH,impl=%s,dim=%d,M=%lu,N=%lu,K=%lu,cycles=%llu,macs=%llu,"
         "ideal_cycles=%llu,util_pct=%llu,result=%s\n",
#if MX_ENABLED
         "mx",
#else
         "stock",
#endif
         DIM, (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k,
         (unsigned long long)cycles, (unsigned long long)macs,
         (unsigned long long)ideal, (unsigned long long)util_pct,
         bad ? "FAIL" : "PASS");
  return bad;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int bad = 0;
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
    bad |= run_shape(&shapes[s]);
  }

  if (bad) {
    printf("mx_bench: FAIL\n");
    exit(1);
  }
  printf("mx_bench: PASS\n");
  exit(0);
}
