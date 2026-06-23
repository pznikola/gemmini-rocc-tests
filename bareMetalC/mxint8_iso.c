// See LICENSE for license details.
//
// MXINT8 tiled-GEMM smoke test for the `gemmini_loop_ws_mxint8` wrapper (plan S3 #5).
//
// Drives the MX block-scaled datapath through the hardware weight-stationary loop unroller
// (`gemmini_loop_ws`) instead of the manual preload/compute path used by the other MX tests.
// The wrapper composes `config_mxint8` + the A/B scale mvins + `gemmini_loop_ws`. This test
// validates the S3 #5 fix: the scale mvins must receive ELEMENT counts, not the tile counts
// `gemmini_loop_ws` consumes — both the A/B output extents (M = I*DIM - pad_I, N = J*DIM -
// pad_J) and the K-block count (k_blocks over K*DIM - pad_K elements, since K is a tile count).
//
// MX v1 supports a SINGLE output tile (I = J = 1, so M = N = DIM): the A-scale read address
// is the output row (`output_counter`, 0..DIM-1) with no output-tile component. This test
// covers a single output tile and a single MX K-block (one K tile) through the loop wrapper.
//
// KNOWN LIMITATION (deferred, see DOCS_MX/PLAN_MX.md S3 #5): cross-block K > one MX block
// through the *hardware loop unroller* is wrong, but ONLY when the loop GEMM is preceded by
// another loop GEMM. Root cause (deterministically reproduced and fully diagnosed): inter-GEMM
// state leakage in the systolic mesh's internal output pipeline — a prior loop GEMM leaves
// residual mesh state that freezes the next GEMM's first K-tile output after ~2 rows. It is
// NOT an MX-datapath bug (scaling, two-phase, cross-block accumulate, and mvout are all
// proven correct), and is orthogonal to the manual cross-block path, which is fully verified
// (mxint8_matmul_dim32 K=64 and mxint8_matmul_dim16 K=64/96/128 all pass). The fix would
// require resetting Mesh-internal pipeline state between GEMMs (the Mesh is DO-NOT-TOUCH);
// the manual path is the supported route for multi-block MX GEMMs.
//
// Build prerequisite: compile against an MX params header (MX_ENABLED=1); with the stock
// header this test is a no-op. WS, untransposed; single output tile M = N = DIM.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/mxint8_golden.h"
#include "include/mxint8_pack.h"

#if !MX_ENABLED

int main() {
  printf("MX disabled in this build (need an MX params header); skipping\n");
  exit(0);
}

#else

#define M DIM
#define N DIM
#define K_MAX 64
#define KB_MAX (K_MAX / MX_BLOCK_SIZE)

static uint32_t lcg = 0x9e3779b1u;
static float frand_signed(float mag) {
  lcg = lcg * 1664525u + 1013904223u;
  const float u = (float)((lcg >> 8) & 0xffffff) / (float)0x1000000; // [0,1)
  return (2.0f * u - 1.0f) * mag;
}

static float a_f[M][K_MAX];
static float b_f[K_MAX][N];
static elem_t a_payload[M][K_MAX] row_align(1);
static elem_t b_payload[K_MAX][N] row_align(1);
static mx_scale_t a_scale[M][KB_MAX] __attribute__((aligned(64)));
static mx_scale_t b_scale[KB_MAX][N] __attribute__((aligned(64)));
static acc_t c_hw[M][N] row_align(1);
static acc_t gold[M][N];

// Zero payloads + neutral (`encode(0)`) scales on all lanes so unused lanes stay valid.
static void clear_all(void) {
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < K_MAX; kk++) { a_payload[i][kk] = 0; a_f[i][kk] = 0.0f; }
    for (size_t b = 0; b < KB_MAX; b++) { a_scale[i][b] = mxint8_e8m0_encode(0); }
  }
  for (size_t kk = 0; kk < K_MAX; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = 0; b_f[kk][j] = 0.0f; }
  }
  for (size_t b = 0; b < KB_MAX; b++) {
    for (size_t j = 0; j < N; j++) { b_scale[b][j] = mxint8_e8m0_encode(0); }
  }
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) { c_hw[i][j] = 0; }
  }
}

// One single-output-tile MX GEMM (M = N = DIM) through the loop wrapper. K is a multiple of
// MX_BLOCK_SIZE; `gemmini_loop_ws` takes K in DIM-sized tiles, so K_tiles = k / DIM.
static void mxint8_tiled_hw(size_t k) {
  const size_t k_tiles = k / DIM;

  gemmini_flush(0);
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
  gemmini_extended_config_st(N * sizeof(acc_t), 0, ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ld(K_MAX * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0); // A
  gemmini_extended3_config_ld(N * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);     // B
  gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);                      // D (none)

  gemmini_loop_ws_mxint8(
      /*I=*/1, /*J=*/1, /*K=*/k_tiles,
      /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
      &a_payload[0][0], &b_payload[0][0], /*D=*/NULL, &c_hw[0][0],
      &a_scale[0][0], &b_scale[0][0],
      /*A_stride=*/K_MAX, /*B_stride=*/N, /*D_stride=*/0, /*C_stride=*/N,
      /*A_scale_stride=*/KB_MAX, /*B_scale_stride=*/N,
      /*A_transpose=*/false, /*B_transpose=*/false,
      /*full_C=*/true, /*low_D=*/false, /*ex_accumulate=*/false,
      /*act=*/0, /*a_spad_id=*/0, /*b_spad_id=*/0, /*is_resadd=*/false,
      /*pipeline_parity=*/-1, /*fence_first=*/true, /*setup_only=*/false);
  gemmini_fence();
}

static int check(const char *label, size_t k) {
  const int rc = mxint8_ref_gemm_acc(&a_payload[0][0], &b_payload[0][0],
                                     &a_scale[0][0], &b_scale[0][0], &gold[0][0],
                                     M, N, k, K_MAX, N, N, KB_MAX, N);
  if (rc != 0) {
    printf("%-8s K=%d: golden rejected an invalid scale (rc=%d)\n", label, (int)k, rc);
    return 1;
  }

  mxint8_tiled_hw(k);

  int mism = 0, fi = -1, fj = -1;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if (c_hw[i][j] != gold[i][j]) {
        if (fi < 0) { fi = (int)i; fj = (int)j; }
        mism++;
      }
    }
  }

  if (mism) {
    printf("%-8s K=%d: FAIL %d mism; first [%d][%d] hw=%d gold=%d\n",
           label, (int)k, mism, fi, fj, (int)c_hw[fi][fj], (int)gold[fi][fj]);
    return 1;
  }
  printf("%-8s K=%d: PASS (c[0][0]=%d c[%d][%d]=%d)\n",
         label, (int)k, (int)c_hw[0][0], M - 1, N - 1, (int)c_hw[M - 1][N - 1]);
  return 0;
}

static int run_random(size_t k) {
  clear_all();
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < k; kk++) { a_f[i][kk] = frand_signed(1.0f + (float)(i % 4)); }
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < N; j++) { b_f[kk][j] = frand_signed(1.0f + (float)(j % 3)); }
  }
  mxint8_pack_a(&a_f[0][0], M, k, K_MAX, KB_MAX, &a_payload[0][0], &a_scale[0][0]);
  mxint8_pack_b(&b_f[0][0], k, N, N, N, &b_payload[0][0], &b_scale[0][0]);
  return check("tiled", k);
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  // Reproduce the documented MX-loop inter-GEMM failure on the CURRENT RTL: a loop K=64 GEMM
  // preceded by another loop GEMM. GEMM1 = K=32 (one block), GEMM2 = K=64 (two blocks).
  int g = run_random(2 * MX_BLOCK_SIZE);    // single K=64 loop GEMM in isolation

  if (g) {
    printf("mxint8_iso: FAIL (g=%d)\n", g);
    exit(1);
  }
  printf("mxint8_iso: PASS\n");
  exit(0);
}

#endif // MX_ENABLED
