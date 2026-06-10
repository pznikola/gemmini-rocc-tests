// Apples-to-apples experiment: stock int8 (NO MXINT8), UNTRANSPOSED weight-stationary, via
// the hardware loop unroller (`gemmini_loop_ws`) -- the exact mesh-feed configuration the MX
// loop wrapper uses, minus the scale sidecar. SHAPE-MATCHED to the failing MX sequence
// (mxint8_btb.c): GEMM1 = K=DIM (one K-tile), then GEMM2 = K=2*DIM (two K-tiles), both
// single-output-tile (M=N=DIM), each checked against a full int32 CPU reference.
//
// 2026-06-10: the first version of this control used K=2*DIM for BOTH GEMMs and PASSED,
// which seemed to exonerate the stock path. But the MX failure is specifically a 1-K-tile
// loop GEMM followed by a 2-K-tile loop GEMM, and the instrumented MX trace shows the
// corruption is the raw mesh output of GEMM2's first K-tile freezing (scales/indices all
// correct). This version reproduces the exact failing K sequence with MX fully out of the
// picture: if GEMM2 fails here too, the freeze is a stock loop-unroller/mesh hand-off
// effect that the MX tests merely expose; if it passes, the trigger is genuinely MX-side.
// RESULT (2026-06-10): PASSES — which pinned the trigger as MX-side and led to the real root
// cause: an RS RAW-hazard miss caused by the scale mvins polluting the RS's CONFIG_LOAD
// decode (fixed in ReservationStation.scala; full story in DOCS_MX/BUG.md).
//
// Config below is copied verbatim from mxint8_btb.c with `config_mxint8` + scale mvins
// removed and `gemmini_loop_ws` called directly.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#define M DIM
#define N DIM
#define K_MAX (2 * DIM)   // max 2 K-tiles, single output tile (I = J = 1)

static elem_t a1[M][K_MAX] row_align(1), b1[K_MAX][N] row_align(1);
static elem_t a2[M][K_MAX] row_align(1), b2[K_MAX][N] row_align(1);
static acc_t  c1[M][N] row_align(1), c2[M][N] row_align(1);
static acc_t  gold[M][N];

static void fill(elem_t *m, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      m[i * cols + j] = (elem_t)((rand() % 3) - 1);
}

static void cpu(elem_t A[M][K_MAX], elem_t B[K_MAX][N], size_t k) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      acc_t s = 0;
      for (size_t kk = 0; kk < k; kk++) s += (acc_t)A[i][kk] * (acc_t)B[kk][j];
      gold[i][j] = s;
    }
}

static void run(elem_t A[M][K_MAX], elem_t B[K_MAX][N], acc_t C[M][N], size_t k) {
  const size_t k_tiles = k / DIM;
  gemmini_flush(0);
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
  gemmini_extended_config_st(N * sizeof(acc_t), 0, ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ld(K_MAX * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0); // A
  gemmini_extended3_config_ld(N * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);     // B
  gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);                      // D (none)

  gemmini_loop_ws(1, 1, k_tiles, 0, 0, 0,
      &A[0][0], &B[0][0], NULL, &C[0][0],
      K_MAX, N, 0, N,
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
  fill((elem_t *)a1, M, K_MAX); fill((elem_t *)b1, K_MAX, N);
  fill((elem_t *)a2, M, K_MAX); fill((elem_t *)b2, K_MAX, N);

  run(a1, b1, c1, DIM);     cpu(a1, b1, DIM);     int m1 = check("GEMM1 untr-loop K=DIM", c1);
  run(a2, b2, c2, 2 * DIM); cpu(a2, b2, 2 * DIM); int m2 = check("GEMM2 untr-loop K=2*DIM", c2);

  int ok = (m1 == 0) && (m2 == 0);
  printf("stock_btb_untr: %s\n", ok ? "PASS" : "FAIL");
  exit(ok ? 0 : 1);
}
