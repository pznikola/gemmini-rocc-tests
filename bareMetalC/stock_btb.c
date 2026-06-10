// Back-to-back stock int8 loop-GEMM experiment (NO MXINT8).
//
// Purpose: test whether a stock Gemmini WS loop GEMM (via tiled_matmul_auto / the
// hardware loop unroller) is corrupted when it immediately follows another loop GEMM --
// i.e. whether the inter-GEMM "mesh output freeze" observed on the MX loop path
// (DOCS_MX/PLAN_MX.md) is a generic stock-Gemmini limitation or MX-specific.
//
// Two consecutive K-tiled GEMMs (M=N=DIM, K=2*DIM => 2 K-tiles, single output tile),
// each compared to a CPU reference. If the freeze is a stock-mesh issue we expect the
// first GEMM to PASS and the second to FAIL (mismatches starting after the first rows).

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#define I_DIM 32
#define J_DIM 32
#define K_DIM 64

static elem_t A1[I_DIM][K_DIM] row_align(1);
static elem_t B1[K_DIM][J_DIM] row_align(1);
static elem_t C1[I_DIM][J_DIM] row_align(1);
static elem_t A2[I_DIM][K_DIM] row_align(1);
static elem_t B2[K_DIM][J_DIM] row_align(1);
static elem_t C2[I_DIM][J_DIM] row_align(1);

static full_t gold_full[I_DIM][J_DIM];

static void fill(elem_t *m, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      m[i * cols + j] = (elem_t)((rand() % 3) - 1); // {-1,0,1}; |C| <= K_DIM < 127, no saturation
}

static void cpu_matmul(elem_t A[I_DIM][K_DIM], elem_t B[K_DIM][J_DIM]) {
  for (int i = 0; i < I_DIM; i++)
    for (int j = 0; j < J_DIM; j++) {
      full_t acc = 0;
      for (int k = 0; k < K_DIM; k++)
        acc += (full_t)A[i][k] * (full_t)B[k][j];
      gold_full[i][j] = acc;
    }
}

static int check(const char *name, elem_t C[I_DIM][J_DIM]) {
  int mism = 0, first_bad_row = -1;
  for (int i = 0; i < I_DIM; i++)
    for (int j = 0; j < J_DIM; j++) {
      elem_t g = (elem_t)gold_full[i][j];
      if (C[i][j] != g) {
        mism++;
        if (first_bad_row < 0) first_bad_row = i;
      }
    }
  printf("%s: %s (mism=%d, first_bad_row=%d) hw[0][0]=%d/g=%d hw[5][0]=%d/g=%d hw[31][0]=%d/g=%d\n",
         name, mism ? "FAIL" : "PASS", mism, first_bad_row,
         C[0][0], (elem_t)gold_full[0][0],
         C[5][0], (elem_t)gold_full[5][0],
         C[31][0], (elem_t)gold_full[31][0]);
  return mism;
}

static void run(elem_t A[I_DIM][K_DIM], elem_t B[K_DIM][J_DIM], elem_t C[I_DIM][J_DIM]) {
  tiled_matmul_auto(I_DIM, J_DIM, K_DIM,
      (elem_t *)A, (elem_t *)B, NULL, (elem_t *)C,
      K_DIM, J_DIM, J_DIM, J_DIM,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
      false, false,
      false, false,
      0,
      WS);
}

int main() {
  gemmini_flush(0);

  fill((elem_t *)A1, I_DIM, K_DIM);
  fill((elem_t *)B1, K_DIM, J_DIM);
  fill((elem_t *)A2, I_DIM, K_DIM);
  fill((elem_t *)B2, K_DIM, J_DIM);

  // GEMM 1 (first loop GEMM -- effectively isolated)
  run(A1, B1, C1);
  cpu_matmul(A1, B1);
  int m1 = check("GEMM1 K=64 (isolated)", C1);

  // GEMM 2 (immediately follows another loop GEMM -- the suspect case)
  run(A2, B2, C2);
  cpu_matmul(A2, B2);
  int m2 = check("GEMM2 K=64 (after GEMM1)", C2);

  int ok = (m1 == 0) && (m2 == 0);
  printf("stock_btb: %s\n", ok ? "PASS" : "FAIL");
  exit(ok ? 0 : 1);
}
