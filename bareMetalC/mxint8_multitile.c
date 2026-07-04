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
// The deterministic probes give every output tile a distinct (1+tile_i)*(1+tile_j)
// signature with row/column-varied scales, so a tile-swap or scale-indexing bug produces
// a diagnosable mismatch pattern rather than noise.
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

static elem_t a_payload[M_MAX][K_MAX] row_align(1);
static elem_t b_payload[K_MAX][N_MAX] row_align(1);
static mx_scale_t a_scale[M_MAX][KB_MAX] __attribute__((aligned(64)));
static mx_scale_t b_scale[KB_MAX][N_MAX] __attribute__((aligned(64)));
static acc_t c_hw[M_MAX][N_MAX] row_align(1);
static acc_t gold[M_MAX][N_MAX];

// Zero payloads + neutral (`encode(0)`) scales on the padded region that the
// current loop can touch, so unused lanes stay valid without clearing the whole
// max-size workspace on the simulated core.
static void clear_all(size_t m, size_t n, size_t k) {
  const size_t m_clear = ((m + DIM - 1) / DIM) * DIM;
  const size_t n_clear = ((n + DIM - 1) / DIM) * DIM;
  const size_t kb_clear = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
  const size_t k_clear = kb_clear * MX_BLOCK_SIZE;

  for (size_t i = 0; i < m_clear; i++) {
    for (size_t kk = 0; kk < k_clear; kk++) { a_payload[i][kk] = 0; }
    for (size_t b = 0; b < kb_clear; b++) { a_scale[i][b] = mxint8_e8m0_encode(0); }
  }
  for (size_t kk = 0; kk < k_clear; kk++) {
    for (size_t j = 0; j < n_clear; j++) { b_payload[kk][j] = 0; }
  }
  for (size_t b = 0; b < kb_clear; b++) {
    for (size_t j = 0; j < n_clear; j++) { b_scale[b][j] = mxint8_e8m0_encode(0); }
  }
  for (size_t i = 0; i < m_clear; i++) {
    for (size_t j = 0; j < n_clear; j++) {
      c_hw[i][j] = 0x7eadbeef;  // poison
      gold[i][j] = 0;
    }
  }
}

static acc_t expected_cell(size_t i, size_t j, size_t k) {
  const size_t kb = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
  int64_t acc = 0;
  for (size_t b = 0; b < kb; b++) {
    const size_t base = b * MX_BLOCK_SIZE;
    const size_t lanes = (base + MX_BLOCK_SIZE <= k) ? MX_BLOCK_SIZE : (k - base);
    const int64_t raw = (int64_t)lanes *
                        (int64_t)a_payload[i][base] *
                        (int64_t)b_payload[base][j];
    acc += mxint8_scale_raw_block(raw,
                                  mxint8_e8m0_decode(a_scale[i][b]),
                                  mxint8_e8m0_decode(b_scale[b][j]));
  }
  return mxint8_saturate_acc(acc);
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
  printf("%-10s %lux%lux%lu: hw begin\n",
         label, (unsigned long)m, (unsigned long)n, (unsigned long)k);
  if (use_tiler) {
    multitile_tiler_hw(m, n, k);
  } else {
    multitile_hw(m, n, k);
  }
  printf("%-10s %lux%lux%lu: compare begin\n",
         label, (unsigned long)m, (unsigned long)n, (unsigned long)k);

  int mism = 0, fi = -1, fj = -1;
  acc_t first_gold = expected_cell(0, 0, k);
  acc_t last_gold = expected_cell(m - 1, n - 1, k);
  const size_t i_tiles = (m + DIM - 1) / DIM;
  const size_t j_tiles = (n + DIM - 1) / DIM;
  for (size_t ti = 0; ti < i_tiles; ti++) {
    const size_t rows[2] = {ti * DIM, ((ti + 1) * DIM < m ? (ti + 1) * DIM : m) - 1};
    for (size_t tj = 0; tj < j_tiles; tj++) {
      const size_t cols[2] = {tj * DIM, ((tj + 1) * DIM < n ? (tj + 1) * DIM : n) - 1};
      for (size_t ri = 0; ri < 2; ri++) {
        for (size_t cj = 0; cj < 2; cj++) {
          const size_t i = rows[ri];
          const size_t j = cols[cj];
          const acc_t exp = expected_cell(i, j, k);
          if (c_hw[i][j] != exp) {
            if (fi < 0) { fi = (int)i; fj = (int)j; }
            mism++;
          }
        }
      }
    }
  }

  if (mism) {
    printf("%-10s %lux%lux%lu: FAIL %d sampled mism; first [%d][%d] (tile %d,%d) hw=%d gold=%d\n",
           label, (unsigned long)m, (unsigned long)n, (unsigned long)k, mism,
           fi, fj, fi / DIM, fj / DIM, (int)c_hw[fi][fj], (int)expected_cell(fi, fj, k));
    return 1;
  }
  printf("%-10s %lux%lux%lu: PASS (sampled tiles, c[0][0]=%d/%d c[%lu][%lu]=%d/%d)\n",
         label, (unsigned long)m, (unsigned long)n, (unsigned long)k,
         (int)c_hw[0][0], (int)first_gold,
         (unsigned long)(m - 1), (unsigned long)(n - 1),
         (int)c_hw[(m - 1)][(n - 1)], (int)last_gold);
  return 0;
}

// Deterministic per-tile probe: payload magnitudes keyed by the output tile, scales
// varied per A row and per B column, so every tile has a unique expected signature.
// Exponents are chosen so the block scale shift is >= -2: the per-tile signatures
// stay NONZERO after rounding (an all-zero output would also "pass" under a wrong
// scale index, which is exactly the bug class the probe exists to expose).
static int run_probe(const char *label, size_t m, size_t n, size_t k, bool use_tiler) {
  printf("%-10s %lux%lux%lu: clear begin\n",
         label, (unsigned long)m, (unsigned long)n, (unsigned long)k);
  clear_all(m, n, k);
  printf("%-10s %lux%lux%lu: clear done\n",
         label, (unsigned long)m, (unsigned long)n, (unsigned long)k);
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

// Structured non-probe case with per-block payload and scale variation. Payloads
// are still constant within each MX block so the target expected-value builder is
// cheap, but the hardware sees multiple output tiles, edge tiles, and K blocks.
static int run_structured(const char *label, size_t m, size_t n, size_t k, bool use_tiler) {
  printf("%-10s %lux%lux%lu: clear begin\n",
         label, (unsigned long)m, (unsigned long)n, (unsigned long)k);
  clear_all(m, n, k);
  printf("%-10s %lux%lux%lu: clear done\n",
         label, (unsigned long)m, (unsigned long)n, (unsigned long)k);
  const size_t kb = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
  for (size_t i = 0; i < m; i++) {
    for (size_t b = 0; b < kb; b++) {
      const size_t base = b * MX_BLOCK_SIZE;
      const size_t end = (base + MX_BLOCK_SIZE <= k) ? base + MX_BLOCK_SIZE : k;
      const elem_t aval = (elem_t)(1 + b + i / DIM);
      for (size_t kk = base; kk < end; kk++) { a_payload[i][kk] = aval; }
      a_scale[i][b] = mxint8_e8m0_encode(5 + (int)((i + b) % 4));
    }
  }
  for (size_t b = 0; b < kb; b++) {
    const size_t base = b * MX_BLOCK_SIZE;
    const size_t end = (base + MX_BLOCK_SIZE <= k) ? base + MX_BLOCK_SIZE : k;
    for (size_t kk = base; kk < end; kk++) {
      for (size_t j = 0; j < n; j++) { b_payload[kk][j] = (elem_t)(1 + j / DIM); }
    }
    for (size_t j = 0; j < n; j++) {
      b_scale[b][j] = mxint8_e8m0_encode(5 + (int)((j + 2 * b) % 3));
    }
  }
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
  bad |= run_structured("edges", 2 * DIM - 3, 2 * DIM - 5, 2 * MX_BLOCK_SIZE, false);
  // Full 2x2 with 4 K blocks (B-scale region exercised across blocks).
  bad |= run_structured("blk2x2", 2 * DIM, 2 * DIM, 4 * MX_BLOCK_SIZE, false);

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
