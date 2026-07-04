// See LICENSE for license details.
//
// MXINT8 back-to-back loop-GEMM regression for the 2026-06-10 ReservationStation RAW-hazard
// fix (see DOCS_MX/BUG.md).
//
// Drives two consecutive GEMMs through the `gemmini_loop_ws_mxint8` wrapper (the hardware WS
// loop unroller): GEMM1 with K = one MX block, then GEMM2 with K = two MX blocks. This exact
// sequence used to corrupt GEMM2's first K-tile: the MX scale mvins (marked `is_config` in the
// reservation station) were wrongly decoded by the RS's CONFIG_LOAD alloc hook, so their DRAM
// pointer bits clobbered the RS's `ld_pixel_repeats`/`ld_block_strides` mirror; the A-payload
// mvin's dependency range was then mis-decoded, the compute's RAW overlap check missed the
// in-flight A mvin, and the compute issued while the A tile was still being DMA-written — the
// mesh was fed stale scratchpad rows (the historical "frozen first K-tile" symptom). The race
// only bit GEMMs whose A mvin was slow (TLB flush + traffic), which made it look "inter-GEMM".
//
// With -DMXINT8_BTB_PROBE=1 (separate binary), GEMM2 uses a deterministic probe whose wrong
// values decode to the exact scale/data the hardware consumed (diagnosis aid).
//
// MX v1 supports a SINGLE output tile (I = J = 1, so M = N = DIM). Build prerequisite: compile
// against an MX params header (MX_ENABLED=1); with the stock header this test is a no-op.
// WS, untransposed.

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

static elem_t a_payload[M][K_MAX] row_align(1);
static elem_t b_payload[K_MAX][N] row_align(1);
static mx_scale_t a_scale[M][KB_MAX] __attribute__((aligned(64)));
static mx_scale_t b_scale[KB_MAX][N] __attribute__((aligned(64)));
static acc_t c_hw[M][N] row_align(1);
static acc_t gold[M][N];

// Zero payloads + neutral (`encode(0)`) scales on all lanes so unused lanes stay valid.
static void clear_all(void) {
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < K_MAX; kk++) { a_payload[i][kk] = 0; }
    for (size_t b = 0; b < KB_MAX; b++) { a_scale[i][b] = mxint8_e8m0_encode(0); }
  }
  for (size_t kk = 0; kk < K_MAX; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = 0; }
  }
  for (size_t b = 0; b < KB_MAX; b++) {
    for (size_t j = 0; j < N; j++) { b_scale[b][j] = mxint8_e8m0_encode(0); }
  }
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      c_hw[i][j] = 0;
      gold[i][j] = 0;
    }
  }
}

static void build_gold_structured(size_t k) {
  const size_t kb = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      int64_t acc = 0;
      for (size_t b = 0; b < kb; b++) {
        const size_t base = b * MX_BLOCK_SIZE;
        const size_t lanes = (base + MX_BLOCK_SIZE <= k) ? MX_BLOCK_SIZE : (k - base);
        const int64_t raw = (int64_t)lanes * (int64_t)(b + 1);
        acc += mxint8_scale_raw_block(raw,
                                      mxint8_e8m0_decode(a_scale[i][b]),
                                      mxint8_e8m0_decode(b_scale[b][j]));
      }
      gold[i][j] = mxint8_saturate_acc(acc);
    }
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

static int run_structured(size_t k) {
  clear_all();
  const size_t kb = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
  for (size_t i = 0; i < M; i++) {
    for (size_t b = 0; b < kb; b++) {
      const size_t base = b * MX_BLOCK_SIZE;
      const size_t end = (base + MX_BLOCK_SIZE <= k) ? base + MX_BLOCK_SIZE : k;
      for (size_t kk = base; kk < end; kk++) { a_payload[i][kk] = (elem_t)(b + 1); }
      a_scale[i][b] = mxint8_e8m0_encode(6 + (int)((i + b) % 4));
    }
  }
  for (size_t b = 0; b < kb; b++) {
    const size_t base = b * MX_BLOCK_SIZE;
    const size_t end = (base + MX_BLOCK_SIZE <= k) ? base + MX_BLOCK_SIZE : k;
    for (size_t kk = base; kk < end; kk++) {
      for (size_t j = 0; j < N; j++) { b_payload[kk][j] = 1; }
    }
    for (size_t j = 0; j < N; j++) {
      b_scale[b][j] = mxint8_e8m0_encode(6 + (int)((j + 2 * b) % 3));
    }
  }
  build_gold_structured(k);
  return check("tiled", k);
}

#ifdef MXINT8_BTB_PROBE
// Deterministic cross-block probe (diagnosis aid, BUG.md §4.1 style): payloads give exact
// power-of-two raw partials (block0 raw = 32/cell, block1 raw = 64/cell) and every
// (row, block) A exponent and (block, col) B exponent is distinct, so a wrong hw value
// decodes directly to WHICH scale entry the hardware actually consumed:
//   c[i][j] = 32 * 2^(eA[i][0]+eB[0][j]-12) + 64 * 2^(eA[i][1]+eB[1][j]-12)
// All shifts land in [-3, +4]; every contribution is exact (no rounding ambiguity).
static int run_probe(size_t k) {
  clear_all();
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < k; kk++) { a_payload[i][kk] = kk < MX_BLOCK_SIZE ? 1 : 2; }
    a_scale[i][0] = mxint8_e8m0_encode(6 + (int)(i % 4));
    a_scale[i][1] = mxint8_e8m0_encode(4 + (int)(i % 3));
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = 1; }
  }
  for (size_t j = 0; j < N; j++) {
    b_scale[0][j] = mxint8_e8m0_encode(6 + (int)(j % 2));
    b_scale[1][j] = mxint8_e8m0_encode(5 + (int)(j % 3));
  }
  build_gold_structured(k);

  const int rc = check("probe", k);
  if (rc) {
    // Compact decode dump: per row, first mismatching column with hw vs gold.
    for (size_t i = 0; i < M; i++) {
      int row_mism = 0, fj = -1;
      for (size_t j = 0; j < N; j++) {
        if (c_hw[i][j] != gold[i][j]) { if (fj < 0) fj = (int)j; row_mism++; }
      }
      if (row_mism)
        printf("  row %2d: %2d mism, first j=%2d hw=%d gold=%d (eA0=%d eA1=%d)\n",
               (int)i, row_mism, fj, (int)c_hw[i][fj], (int)gold[i][fj],
               6 + (int)(i % 4), 4 + (int)(i % 3));
    }
  }
  return rc;
}
#endif // MXINT8_BTB_PROBE

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  // Reproduce the documented MX-loop inter-GEMM failure on the CURRENT RTL: a loop K=64 GEMM
  // preceded by another loop GEMM. GEMM1 = K=32 (one block), GEMM2 = K=64 (two blocks).
  // With -DMXINT8_BTB_PROBE=1, GEMM2 uses the deterministic probe instead, whose wrong values
  // decode to the exact scale entry the hardware consumed (separate binary so the random
  // repro's instruction stream stays byte-identical -- the bug is timing-sensitive).
  int g1 = run_structured(MX_BLOCK_SIZE);       // GEMM1: K=32
#ifdef MXINT8_BTB_PROBE
  int g2 = run_probe(2 * MX_BLOCK_SIZE);    // GEMM2: deterministic K=64, preceded by GEMM1
#else
  int g2 = run_structured(2 * MX_BLOCK_SIZE);   // GEMM2: K=64, preceded by GEMM1
#endif

  if (g1 || g2) {
    printf("mxint8_btb: FAIL (g1=%d g2=%d)\n", g1, g2);
    exit(1);
  }
  printf("mxint8_btb: PASS\n");
  exit(0);
}

#endif // MX_ENABLED
