// See LICENSE for license details.
//
// MXINT8 N-phase single-tile GEMM test (Stage B, DIM generalization).
//
// DIM-agnostic: reads DIM and MX_BLOCK_SIZE from the staged params header, so the same
// source exercises the N = MX_BLOCK_SIZE/DIM phase count at whatever DIM was built —
// 1 phase at DIM=32, 2 at DIM=16, 4 at DIM=8, 8 at DIM=4. When DIM < MX_BLOCK_SIZE each
// logical 32-K block is fed as N physical phases; the ExecuteController accumulates the
// running raw partial across the non-final phases into one DIM x DIM buffer and applies
// the per-block scale once on the last phase. This test drives that path through the
// single-tile loop wrapper (`gemmini_loop_ws_mxint8`, I=J=1) and diffs against the
// bit-exact golden `mxint8_ref_gemm_acc`.
//
// Cases (single output tile M = N = DIM):
//   - probe K=1 block: per-row/col E8M0 scales give each output a distinct nonzero
//     signature; an exact match proves all N phases summed into the full 32-lane block
//     before the single scale (a dropped phase would scale a partial sum).
//   - random K = 1..4 blocks: packer-quantized fp32, exercises multi-block accumulation
//     (cross-block scale-then-accumulate) on top of the intra-block phase accumulation.
//
// Build prerequisite: an MX params header (MX_ENABLED=1). With the stock header this is
// a no-op. WS, untransposed.

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
#define K_MAX (4 * MX_BLOCK_SIZE)
#define KB_MAX (K_MAX / MX_BLOCK_SIZE)

static uint32_t lcg = 0x1234567u;
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
    for (size_t j = 0; j < N; j++) { c_hw[i][j] = 0x7eadbeef; }  // poison
  }
}

// One single-output-tile MX GEMM (M = N = DIM) through the loop wrapper. K is a multiple
// of MX_BLOCK_SIZE; gemmini_loop_ws takes K in DIM-sized tiles, so K_tiles = k / DIM.
static void nphase_hw(size_t k) {
  gemmini_flush(0);
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
  gemmini_extended_config_st(N * sizeof(acc_t), 0, ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ld(K_MAX * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0); // A
  gemmini_extended3_config_ld(N * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);     // B
  gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);                      // D

  gemmini_loop_ws_mxint8(
      /*I=*/1, /*J=*/1, /*K=*/k / DIM,
      /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
      &a_payload[0][0], &b_payload[0][0], /*D=*/NULL, &c_hw[0][0],
      &a_scale[0][0], &b_scale[0][0],
      /*A_stride=*/K_MAX, /*B_stride=*/N, /*D_stride=*/0, /*C_stride=*/N,
      /*A_scale_stride=*/KB_MAX, /*B_scale_stride=*/N,
      /*A_transpose=*/false, /*B_transpose=*/false,
      /*full_C=*/true, /*low_D=*/false, /*ex_accumulate=*/false,
      /*act=*/0, /*a_spad_id=*/0, /*b_spad_id=*/0, /*is_resadd=*/false);
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
  nphase_hw(k);
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
    printf("%-8s K=%d (DIM=%d, %d phases/block): FAIL %d mism; first [%d][%d] hw=%d gold=%d\n",
           label, (int)k, DIM, MX_BLOCK_SIZE / DIM, mism, fi, fj,
           (int)c_hw[fi][fj], (int)gold[fi][fj]);
    return 1;
  }
  printf("%-8s K=%d (DIM=%d, %d phases/block): PASS (c[0][0]=%d c[%d][%d]=%d)\n",
         label, (int)k, DIM, MX_BLOCK_SIZE / DIM, (int)c_hw[0][0],
         M - 1, N - 1, (int)c_hw[M - 1][N - 1]);
  return 0;
}

// Deterministic probe: payloads = 2, per-row A scale and per-block/col B scale chosen so
// each output is a distinct nonzero value spanning a few scale shifts. Each of the N
// physical phases contributes DIM lanes of 2*2=4; the full block sums 32*4=128, scaled.
static int run_probe(size_t k) {
  clear_all();
  const size_t kb = k / MX_BLOCK_SIZE;
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < k; kk++) { a_payload[i][kk] = 2; }
    for (size_t b = 0; b < kb; b++) { a_scale[i][b] = mxint8_e8m0_encode((int)(i % 3) + 5); }
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = 2; }
  }
  for (size_t b = 0; b < kb; b++) {
    for (size_t j = 0; j < N; j++) { b_scale[b][j] = mxint8_e8m0_encode((int)(j % 2) + (int)b + 5); }
  }
  return check("probe", k);
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
  return check("random", k);
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int bad = 0;
  bad |= run_probe(MX_BLOCK_SIZE);          // 1 block (N phases)
  bad |= run_random(MX_BLOCK_SIZE);         // 1 block, random
  bad |= run_random(2 * MX_BLOCK_SIZE);     // 2 blocks (cross-block accumulate)
  bad |= run_probe(3 * MX_BLOCK_SIZE);      // 3 blocks, per-block-distinct probe
  bad |= run_random(4 * MX_BLOCK_SIZE);     // 4 blocks, random

  if (bad) {
    printf("mxint8_matmul_nphase: FAIL\n");
    exit(1);
  }
  printf("mxint8_matmul_nphase: PASS\n");
  exit(0);
}

#endif // MX_ENABLED
