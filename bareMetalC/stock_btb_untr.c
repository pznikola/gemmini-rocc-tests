// Apples-to-apples experiment: stock int8 (NO MXINT8), UNTRANSPOSED weight-stationary, via
// the hardware loop unroller (`gemmini_loop_ws`) -- the exact mesh-feed configuration the MX
// loop wrapper uses, minus the scale sidecar. Two back-to-back single-output-tile GEMMs
// (M=N=DIM, K=2*DIM => 2 K-tiles), each compared to a full int32 CPU reference.
//
// The standard `tiled_matmul_auto` WS path feeds B through the transposer; MX requires
// untransposed WS, which is the path that exhibited the inter-GEMM "mesh output freeze"
// (DOCS_MX/PLAN_MX.md). If that freeze is the untransposed-loop path itself (a stock-Gemmini
// corner), GEMM2 should be corrupted here even with no MX. If both pass, the failure is
// specific to the MX wrapper/scaling, not the bare untransposed loop.
//
// Config below is copied verbatim from the known-working MX loop test (mxint8_tiled.c),
// with the `config_mxint8` + scale mvins removed and `gemmini_loop_ws` called directly.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#define M DIM
#define N DIM
#define K (2 * DIM)   // 2 K-tiles, single output tile (I = J = 1)

static elem_t a1[M][K] row_align(1), b1[K][N] row_align(1);
static elem_t a2[M][K] row_align(1), b2[K][N] row_align(1);
static acc_t  c1[M][N] row_align(1), c2[M][N] row_align(1);
static acc_t  gold[M][N];

static void fill(elem_t *m, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      m[i * cols + j] = (elem_t)((rand() % 3) - 1);
}

static void cpu(elem_t A[M][K], elem_t B[K][N]) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      acc_t s = 0;
      for (int k = 0; k < K; k++) s += (acc_t)A[i][k] * (acc_t)B[k][j];
      gold[i][j] = s;
    }
}

static void run(elem_t A[M][K], elem_t B[K][N], acc_t C[M][N]) {
  const size_t k_tiles = K / DIM;
  gemmini_flush(0);
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
  gemmini_extended_config_st(N * sizeof(acc_t), 0, ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ld(K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0); // A
  gemmini_extended3_config_ld(N * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1); // B
  gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);                  // D (none)

  gemmini_loop_ws(1, 1, k_tiles, 0, 0, 0,
      &A[0][0], &B[0][0], NULL, &C[0][0],
      K, N, 0, N,
      /*A_transpose=*/false, /*B_transpose=*/false,
      /*full_C=*/true, /*low_D=*/false, /*ex_accumulate=*/false,
      /*act=*/0, /*a_spad_id=*/0, /*b_spad_id=*/0, /*is_resadd=*/false);
  gemmini_fence();
}

static int check(const char *name, acc_t C[M][N]) {
  int mism = 0, first_bad_row = -1;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      if (C[i][j] != gold[i][j]) { mism++; if (first_bad_row < 0) first_bad_row = i; }
  printf("%s: %s (mism=%d, first_bad_row=%d) hw[0][0]=%d/g=%d hw[%d][0]=%d/g=%d\n",
         name, mism ? "FAIL" : "PASS", mism, first_bad_row,
         (int)C[0][0], (int)gold[0][0], M - 1, (int)C[M - 1][0], (int)gold[M - 1][0]);
  return mism;
}

int main() {
  gemmini_flush(0);
  fill((elem_t *)a1, M, K); fill((elem_t *)b1, K, N);
  fill((elem_t *)a2, M, K); fill((elem_t *)b2, K, N);

  run(a1, b1, c1); cpu(a1, b1); int m1 = check("GEMM1 untr-loop (isolated)", c1);
  run(a2, b2, c2); cpu(a2, b2); int m2 = check("GEMM2 untr-loop (after GEMM1)", c2);

  int ok = (m1 == 0) && (m2 == 0);
  printf("stock_btb_untr: %s\n", ok ? "PASS" : "FAIL");
  exit(ok ? 0 : 1);
}
