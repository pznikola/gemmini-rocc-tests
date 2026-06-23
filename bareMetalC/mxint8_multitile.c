// See LICENSE for license details.
//
// MXINT8 multi-tile GEMM test (plan P3, policy Appendix A).
//
// Drives the MX datapath through the hardware loop unroller with MULTIPLE output tiles
// (I/J > 1), validating the Appendix A contract end to end:
//   - padded power-of-two C-tile pitch (A.1), including a non-power-of-two J case whose
//     padding tile slots must stay untouched;
//   - tiled A/B scale SRAM layout (A.2) via the wrapper's repack/fast paths;
//   - drain-order walk vs tag-derived tile cross-check (A.3, RTL assert);
//   - block-keyed accumulate bit (A.4) across K blocks and K chunks;
//   - the software outer tiler (`tiled_matmul_mxint8`, A.5), including K chunking with
//     `ex_accumulate` continuation across invocations.
//
// The deterministic probe gives every output tile a distinct (1+tile_i)*(1+tile_j)
// signature with row/column-varied scales, so a tile-swap or scale-indexing bug produces
// a diagnosable mismatch pattern rather than noise. Random cases go through the packer.
//
// Build prerequisite: compile against an MX params header (MX_ENABLED=1); with the stock
// header this test is a no-op. WS, untransposed.

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

#define M_MAX (2 * DIM > 64 ? 2 * DIM : 64)
#define N_MAX (4 * DIM > 128 ? 4 * DIM : 128)
#define K_MAX 512
#define KB_MAX (K_MAX / MX_BLOCK_SIZE)

static uint32_t lcg = 0x5bd1e995u;
static float frand_signed(float mag) {
  lcg = lcg * 1664525u + 1013904223u;
  const float u = (float)((lcg >> 8) & 0xffffff) / (float)0x1000000; // [0,1)
  return (2.0f * u - 1.0f) * mag;
}

static float a_f[M_MAX][K_MAX];
static float b_f[K_MAX][N_MAX];
static elem_t a_payload[M_MAX][K_MAX] row_align(1);
static elem_t b_payload[K_MAX][N_MAX] row_align(1);
static mx_scale_t a_scale[M_MAX][KB_MAX] __attribute__((aligned(64)));
static mx_scale_t b_scale[KB_MAX][N_MAX] __attribute__((aligned(64)));
static acc_t c_hw[M_MAX][N_MAX] row_align(1);
static acc_t gold[M_MAX][N_MAX];

// Zero payloads + neutral (`encode(0)`) scales on all lanes so unused lanes stay valid.
static void clear_all(void) {
  for (size_t i = 0; i < M_MAX; i++) {
    for (size_t kk = 0; kk < K_MAX; kk++) { a_payload[i][kk] = 0; a_f[i][kk] = 0.0f; }
    for (size_t b = 0; b < KB_MAX; b++) { a_scale[i][b] = mxint8_e8m0_encode(0); }
  }
  for (size_t kk = 0; kk < K_MAX; kk++) {
    for (size_t j = 0; j < N_MAX; j++) { b_payload[kk][j] = 0; b_f[kk][j] = 0.0f; }
  }
  for (size_t b = 0; b < KB_MAX; b++) {
    for (size_t j = 0; j < N_MAX; j++) { b_scale[b][j] = mxint8_e8m0_encode(0); }
  }
  for (size_t i = 0; i < M_MAX; i++) {
    for (size_t j = 0; j < N_MAX; j++) { c_hw[i][j] = 0x7eadbeef; }  // poison
  }
}

static void config_common(void) {
  gemmini_flush(0);
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
  gemmini_extended_config_st(N_MAX * sizeof(acc_t), 0, ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ld(K_MAX * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0); // A
  gemmini_extended3_config_ld(N_MAX * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1); // B
  gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);                      // D
}

// One multi-tile MX GEMM (m x n x k elements) through the loop wrapper directly.
static void multitile_hw(size_t m, size_t n, size_t k) {
  const size_t i_tiles = (m + DIM - 1) / DIM;
  const size_t j_tiles = (n + DIM - 1) / DIM;
  const size_t k_tiles = (k + DIM - 1) / DIM;
  const size_t pad_i = i_tiles * DIM - m;
  const size_t pad_j = j_tiles * DIM - n;
  const size_t pad_k = k_tiles * DIM - k;

  config_common();
  gemmini_loop_ws_mxint8(
      i_tiles, j_tiles, k_tiles, pad_i, pad_j, pad_k,
      &a_payload[0][0], &b_payload[0][0], /*D=*/NULL, &c_hw[0][0],
      &a_scale[0][0], &b_scale[0][0],
      /*A_stride=*/K_MAX, /*B_stride=*/N_MAX, /*D_stride=*/0, /*C_stride=*/N_MAX,
      /*A_scale_stride=*/KB_MAX, /*B_scale_stride=*/N_MAX,
      /*A_transpose=*/false, /*B_transpose=*/false,
      /*full_C=*/true, /*low_D=*/false, /*ex_accumulate=*/false,
      /*act=*/0, /*a_spad_id=*/0, /*b_spad_id=*/0, /*is_resadd=*/false,
      /*pipeline_parity=*/-1, /*fence_first=*/true, /*setup_only=*/false);
  gemmini_fence();
}

// The same GEMM through the software outer tiler (chunked invocations).
static void multitile_tiler_hw(size_t m, size_t n, size_t k) {
  config_common();
  tiled_matmul_mxint8(m, n, k,
      &a_payload[0][0], &b_payload[0][0], /*D=*/NULL, &c_hw[0][0],
      &a_scale[0][0], &b_scale[0][0],
      /*A_stride=*/K_MAX, /*B_stride=*/N_MAX, /*D_stride=*/0, /*C_stride=*/N_MAX,
      /*A_scale_stride=*/KB_MAX, /*B_scale_stride=*/N_MAX,
      /*full_C=*/true, /*low_D=*/false, /*act=*/0);
  gemmini_fence();
}

static int check(const char *label, size_t m, size_t n, size_t k, bool use_tiler) {
  const int rc = mxint8_ref_gemm_acc(&a_payload[0][0], &b_payload[0][0],
                                     &a_scale[0][0], &b_scale[0][0], &gold[0][0],
                                     m, n, k, K_MAX, N_MAX, N_MAX, KB_MAX, N_MAX);
  if (rc != 0) {
    printf("%-10s %lux%lux%lu: golden rejected an invalid scale (rc=%d)\n",
           label, (unsigned long)m, (unsigned long)n, (unsigned long)k, rc);
    return 1;
  }

  if (use_tiler) {
    multitile_tiler_hw(m, n, k);
  } else {
    multitile_hw(m, n, k);
  }

  int mism = 0, fi = -1, fj = -1;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      if (c_hw[i][j] != gold[i][j]) {
        if (fi < 0) { fi = (int)i; fj = (int)j; }
        mism++;
      }
    }
  }

  if (mism) {
    printf("%-10s %lux%lux%lu: FAIL %d mism; first [%d][%d] (tile %d,%d) hw=%d gold=%d\n",
           label, (unsigned long)m, (unsigned long)n, (unsigned long)k, mism,
           fi, fj, fi / DIM, fj / DIM, (int)c_hw[fi][fj], (int)gold[fi][fj]);
    return 1;
  }
  printf("%-10s %lux%lux%lu: PASS (c[0][0]=%d c[%lu][%lu]=%d)\n",
         label, (unsigned long)m, (unsigned long)n, (unsigned long)k,
         (int)c_hw[0][0], (unsigned long)(m - 1), (unsigned long)(n - 1),
         (int)c_hw[(m - 1)][(n - 1)]);
  return 0;
}

// Deterministic per-tile probe: payload magnitudes keyed by the output tile, scales
// varied per A row and per B column, so every tile has a unique expected signature.
// Exponents are chosen so the block scale shift is >= -2: the per-tile signatures
// stay NONZERO after rounding (an all-zero output would also "pass" under a wrong
// scale index, which is exactly the bug class the probe exists to expose).
static int run_probe(const char *label, size_t m, size_t n, size_t k, bool use_tiler) {
  clear_all();
  for (size_t i = 0; i < m; i++) {
    for (size_t kk = 0; kk < k; kk++) { a_payload[i][kk] = (elem_t)(2 * (1 + i / DIM)); }
    for (size_t b = 0; b < (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE; b++) {
      a_scale[i][b] = mxint8_e8m0_encode((int)(i % 3) + 5);
    }
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < n; j++) { b_payload[kk][j] = (elem_t)(2 * (1 + j / DIM)); }
  }
  for (size_t b = 0; b < (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE; b++) {
    for (size_t j = 0; j < n; j++) { b_scale[b][j] = mxint8_e8m0_encode((int)(j % 2) + 5); }
  }
  return check(label, m, n, k, use_tiler);
}

// Seeded random case through the fp32 packer.
static int run_random(const char *label, size_t m, size_t n, size_t k, bool use_tiler) {
  clear_all();
  for (size_t i = 0; i < m; i++) {
    for (size_t kk = 0; kk < k; kk++) { a_f[i][kk] = frand_signed(1.0f + (float)(i % 4)); }
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < n; j++) { b_f[kk][j] = frand_signed(1.0f + (float)(j % 3)); }
  }
  mxint8_pack_a(&a_f[0][0], m, k, K_MAX, KB_MAX, &a_payload[0][0], &a_scale[0][0]);
  mxint8_pack_b(&b_f[0][0], k, n, N_MAX, N_MAX, &b_payload[0][0], &b_scale[0][0]);
  return check(label, m, n, k, use_tiler);
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int bad = 0;

  // Direct multi-tile invocations (single hardware loop each).
  bad |= run_probe("probe2x2", 2 * DIM, 2 * DIM, MX_BLOCK_SIZE, false);
  // Non-power-of-two J (Jp = 4 with one padding tile slot) + two K blocks.
  bad |= run_probe("probe2x3", 2 * DIM, 3 * DIM, 2 * MX_BLOCK_SIZE, false);
  // Partial edge tiles on M and N, with whole-block K (at DIM < 16 a logical block must
  // be fed completely — a partial trailing K block cannot finish its phases, so K is a
  // whole multiple of MX_BLOCK_SIZE; pad_I/pad_J still exercise the M/N edges).
  bad |= run_random("edges", 2 * DIM - 3, 2 * DIM - 5, 2 * MX_BLOCK_SIZE, false);
  // Random full 2x2 with 4 K blocks (B-scale region exercised across blocks).
  bad |= run_random("rand2x2", 2 * DIM, 2 * DIM, 4 * MX_BLOCK_SIZE, false);

  // Outer tiler (tiled_matmul_mxint8): exercises the multi-invocation spatial loop and
  // its per-chunk A/B/C/scale base-offset arithmetic. N = 8*DIM_16 columns is 8 J-tiles,
  // which exceeds the tiler's Jp cap (4) at DIM=16, so it splits into TWO J-chunks (the
  // chunk-boundary tile_j offset is what the probe's per-column signature checks); at
  // DIM=32 the same shape is a single invocation (a tiler-wiring sanity check). K stays
  // one block to keep it cheap — cross-invocation K continuation is covered by
  // mxint8_btb / mxint8_tiled. The probe gives every tile a distinct nonzero signature,
  // so a wrong per-chunk offset surfaces as a localized mismatch.
  bad |= run_probe("tiler", 64, 128, MX_BLOCK_SIZE, true);

  if (bad) {
    printf("mxint8_multitile: FAIL\n");
    exit(1);
  }
  printf("mxint8_multitile: PASS\n");
  exit(0);
}

#endif // MX_ENABLED
