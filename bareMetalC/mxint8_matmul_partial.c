// See LICENSE for license details.
//
// MXINT8 DIM=32 partial output tiles (M <= DIM and/or N <= DIM).
//
// Closes the last DIM=32 gap: the earlier mxint8 tests only exercise a full
// M = N = DIM output tile. Here the payload mvin, the preload/compute, and the
// mvout all carry partial row/column counts (the extended-dim intrinsics), so
// the hardware must mask the output to the valid M x N region while still
// applying the per-row A scale and per-column B scale correctly.
//
// Partial-tile scale contract (no hardware change required):
//   * The payload mvin, compute, and mvout use the real M (rows) and N (cols).
//   * The MX scale mvin always covers the FULL DIM lanes/rows, padding the
//     unused rows (A) and columns (B) with the neutral scale `encode(0)`. This
//     is required because (a) the B-scale validity assert checks all DIM lanes
//     of the loaded vector and (b) the A-scale read counter wraps at the mesh
//     `total_rows` (= K block size = DIM here), so every DIM-th row is read even
//     when only M < DIM rows are written out. The output masking
//     (`output_counter < C_rows`, `lane < C_cols`) drops the padded rows/cols.
//
// Build prerequisite: compile against gemmini_params_mxint8_dim32.h
// (MX_ENABLED=1); with the stock header this test is a no-op.

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
  printf("MX disabled in this build (need gemmini_params_mxint8_dim32.h); skipping\n");
  exit(0);
}

#else

#define M_MAX DIM
#define N_MAX DIM
#define K_MAX 64
#define KB_MAX (K_MAX / MX_BLOCK_SIZE)

static elem_t a_payload[M_MAX][K_MAX] row_align(1);
static elem_t b_payload[K_MAX][N_MAX] row_align(1);
static mx_scale_t a_scale[M_MAX][KB_MAX] __attribute__((aligned(64)));
static mx_scale_t b_scale[KB_MAX][N_MAX] __attribute__((aligned(64)));
static acc_t c_hw[M_MAX][N_MAX] row_align(1);
static acc_t gold[M_MAX][N_MAX];

// Reset every buffer to a well-defined background: zero payloads (so the golden,
// which only reads the M x N region, is unaffected) and neutral scales on ALL
// DIM rows/cols (so the padded scale lanes loaded into MXScaleSRAM are valid).
static void clear_all(void) {
  for (size_t i = 0; i < M_MAX; i++) {
    for (size_t kk = 0; kk < K_MAX; kk++) { a_payload[i][kk] = 0; }
    for (size_t b = 0; b < KB_MAX; b++) { a_scale[i][b] = mxint8_e8m0_encode(0); }
  }
  for (size_t kk = 0; kk < K_MAX; kk++) {
    for (size_t j = 0; j < N_MAX; j++) { b_payload[kk][j] = 0; }
  }
  for (size_t b = 0; b < KB_MAX; b++) {
    for (size_t j = 0; j < N_MAX; j++) { b_scale[b][j] = mxint8_e8m0_encode(0); }
  }
  for (size_t i = 0; i < M_MAX; i++) {
    for (size_t j = 0; j < N_MAX; j++) { c_hw[i][j] = 0; }
  }
}

static void build_gold_structured(size_t m, size_t n, size_t k, size_t kb) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
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

// One MXINT8 GEMM producing an M x N output tile (M, N may be < DIM). Payload
// mvin / preload / compute / mvout all use the partial dims; the K dimension of
// each MX block is full (MX_BLOCK_SIZE == DIM), so only M and N are partial.
static void mxint8_hw_matmul(size_t m, size_t n, size_t k, size_t kb) {
  const uint32_t acc_base = (1u << (ADDR_LEN - 1)) | (1u << (ADDR_LEN - 3)); // is_acc + read_full
  const uint32_t acc_accum = (1u << (ADDR_LEN - 2));                         // accumulate-on-write
  const uint32_t a_sp = 0;
  const uint32_t b_sp = KB_MAX * DIM;
  (void)k;

  gemmini_flush(0);
  gemmini_config_mxint8(true, 0);
  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);
  gemmini_config_st(N_MAX * sizeof(acc_t));
  // Scales: full DIM rows (A) and full DIM cols (B); padding lanes are neutral.
  gemmini_mvin_mxscale_a(&a_scale[0][0], 0, DIM, kb, KB_MAX);
  gemmini_mvin_mxscale_b(&b_scale[0][0], MX_SCALE_SP_ROWS / 2, kb, DIM, N_MAX);

  // Payloads: A is m x K (m partial rows), B is K x n (n partial cols).
  gemmini_config_ld(K_MAX * sizeof(elem_t));
  for (size_t b = 0; b < kb; b++) {
    gemmini_extended_mvin(&a_payload[0][b * MX_BLOCK_SIZE], a_sp + b * DIM, DIM, m);
  }
  gemmini_config_ld(N_MAX * sizeof(elem_t));
  for (size_t b = 0; b < kb; b++) {
    gemmini_extended_mvin(&b_payload[b * MX_BLOCK_SIZE][0], b_sp + b * DIM, n, DIM);
  }

  for (size_t b = 0; b < kb; b++) {
    const uint32_t c_addr = acc_base | (b == 0 ? 0u : acc_accum);
    // preload B (BD_cols=n, BD_rows=K=DIM), output tile C_cols=n, C_rows=m.
    gemmini_extended_preload(b_sp + b * DIM, c_addr, n, DIM, n, m);
    // compute A (A_cols=K=DIM, A_rows=m); the garbage BD operand stays full DIM.
    gemmini_extended_compute_preloaded(a_sp + b * DIM, GARBAGE_ADDR, DIM, m, DIM, DIM);
  }

  gemmini_extended_mvout(&c_hw[0][0], acc_base, n, m);
  gemmini_fence();
}

// Golden + hardware diff over the M x N region for the data currently staged.
static int check(const char *label, size_t m, size_t n, size_t k, size_t kb) {
  mxint8_hw_matmul(m, n, k, kb);

  int mism = 0, fi = -1, fj = -1;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      if (c_hw[i][j] != gold[i][j]) {
        if (fi < 0) { fi = (int)i; fj = (int)j; }
        mism++;
      }
    }
  }
  // Also confirm the hardware did not write outside the valid M x N region.
  int spill = 0;
  for (size_t i = 0; i < M_MAX; i++) {
    for (size_t j = 0; j < N_MAX; j++) {
      if ((i >= m || j >= n) && c_hw[i][j] != 0) { spill++; }
    }
  }

  if (mism || spill) {
    printf("%-10s M=%d N=%d K=%d: FAIL %d mism, %d spill; first [%d][%d] hw=%d gold=%d\n",
           label, (int)m, (int)n, (int)k, mism, spill, fi, fj,
           (fi < 0 ? 0 : (int)c_hw[fi][fj]), (fi < 0 ? 0 : (int)gold[fi][fj]));
    return 1;
  }
  printf("%-10s M=%d N=%d K=%d: PASS (c[0][0]=%d c[%d][%d]=%d)\n",
         label, (int)m, (int)n, (int)k, (int)c_hw[0][0],
         (int)m - 1, (int)n - 1, (int)c_hw[m - 1][n - 1]);
  return 0;
}

// Deterministic probe over a partial tile: all-ones payloads, eA[i]=6+(i%4),
// eB[j]=6+(j%3) ⇒ hw[i][j] = K << ((i%4)+(j%3)). Pins the per-output-row A-scale
// and per-output-column B-scale addressing when M, N < DIM.
static int run_probe(size_t m, size_t n) {
  clear_all();
  const size_t k = MX_BLOCK_SIZE;
  for (size_t i = 0; i < m; i++) {
    for (size_t kk = 0; kk < k; kk++) { a_payload[i][kk] = 1; }
    a_scale[i][0] = mxint8_e8m0_encode(6 + (int)(i % 4));
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < n; j++) { b_payload[kk][j] = 1; }
  }
  for (size_t j = 0; j < n; j++) { b_scale[0][j] = mxint8_e8m0_encode(6 + (int)(j % 3)); }
  build_gold_structured(m, n, k, 1);
  return check("probe", m, n, k, 1);
}

// Deterministic structured case with varied row/column scales. K may span two MX
// blocks to combine partial M/N with cross-block accumulation, while expected
// results stay cheap to build inside the simulated core.
static int run_structured(const char *label, size_t m, size_t n, size_t k) {
  clear_all();
  const size_t kb = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
  for (size_t i = 0; i < m; i++) {
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
      for (size_t j = 0; j < n; j++) { b_payload[kk][j] = 1; }
    }
    for (size_t j = 0; j < n; j++) {
      b_scale[b][j] = mxint8_e8m0_encode(6 + (int)((j + 2 * b) % 3));
    }
  }
  build_gold_structured(m, n, k, kb);
  return check(label, m, n, k, kb);
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int bad = 0;
  bad |= run_probe(10, 12);                  // partial probe: per-row/col addressing
  bad |= run_structured("full",  M_MAX, N_MAX, MX_BLOCK_SIZE);  // extended dims == full tile
  bad |= run_structured("row1",  1,     N_MAX, MX_BLOCK_SIZE);  // single output row
  bad |= run_structured("col1",  M_MAX, 1,     MX_BLOCK_SIZE);  // single output column
  bad |= run_structured("part",  13,    20,    MX_BLOCK_SIZE);  // general partial tile
  bad |= run_structured("part2", 7,     9,     2 * MX_BLOCK_SIZE); // partial + cross-block K

  if (bad) {
    printf("mxint8_matmul_partial: FAIL\n");
    exit(1);
  }
  printf("mxint8_matmul_partial: PASS\n");
  exit(0);
}

#endif // MX_ENABLED
