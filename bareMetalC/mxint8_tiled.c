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
// covers a single output tile with K = one and two MX K-blocks through the loop wrapper,
// run back-to-back (the second GEMM following the first in the same program).
//
// HISTORY (fixed 2026-06-10, see DOCS_MX/BUG.md): cross-block K through the loop wrapper used
// to fail when the loop GEMM was preceded by another loop GEMM. The real root cause was a
// ReservationStation RAW-hazard miss: the MX scale mvins (marked `is_config`) were wrongly
// decoded by the RS's CONFIG_LOAD alloc hook, so their DRAM pointer bits clobbered the RS's
// `ld_pixel_repeats`/`ld_block_strides` mirror; the following A-payload mvin's dependency
// range was then mis-decoded, the compute issued while the A tile was still being DMA-written,
// and the mesh was fed stale scratchpad rows (the historical "frozen first K-tile" symptom).
// Fixed by excluding `is_mx_scale` entries from that hook (ReservationStation.scala); the
// mesh, MX datapath, and loop unroller were all correct. The back-to-back K=32 -> K=64 case
// below is the regression for that fix (also covered by bareMetalC/mxint8_btb.c).
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

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int bad = 0;
  bad |= run_structured(MX_BLOCK_SIZE);      // single MX block (one K tile) through the loop wrapper
  // Cross-block K back-to-back after another loop GEMM: regression for the 2026-06-10
  // ReservationStation RAW-hazard fix (see the header note and DOCS_MX/BUG.md).
  bad |= run_structured(2 * MX_BLOCK_SIZE);  // two MX blocks (two K tiles), preceded by the GEMM above

  if (bad) {
    printf("mxint8_tiled: FAIL\n");
    exit(1);
  }
  printf("mxint8_tiled: PASS\n");
  exit(0);
}

#endif // MX_ENABLED
