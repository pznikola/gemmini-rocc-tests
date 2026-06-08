// See LICENSE for license details.
//
// MXINT8 DIM=16 two-phase hardware GEMM test (the research contribution).
//
// At DIM=16 the physical systolic array (16 wide) is narrower than the logical
// OCP MX block (32 K-elements), so one logical block spans TWO physical K phases.
// The hardware (the inline BlockScaleUnit in ExecuteController) buffers the raw
// int partials of the first 16-wide half (`mx_raw_half`, accumulator write
// suppressed via `mx_suppress_acc_write`), then on the second half sums both
// halves' raw partials and applies the per-block scale 2^(eA+eB-12) exactly ONCE
// before accumulating. The A/B block scales are held constant across both halves.
//
// This test drives that path from software: each logical 32-K block is issued as
// two preload(B half)/compute(A half) pairs of 16 K-rows, and the result is
// diffed against the bit-exact software golden `mxint8_ref_gemm_acc` (which sums
// the full 32-K block, then scales once). An exact match proves the two-phase raw
// buffering and the once-per-logical-block scaling.
//
// Cases (all single logical block, K = MX_BLOCK_SIZE = 32 = two 16-wide phases):
//   1. A deterministic "probe": all-ones payloads with per-row/col scales chosen
//      so hw[i][j] = K << ((i%4)+(j%3)). Since the K=32 raw sum is accumulated
//      from two 16-wide halves (16+16), an exact match directly distinguishes a
//      correct two-phase sum from a broken one (which would yield 16<<shift).
//   2/3. Randomized cases packed by `mxint8_pack.h` (two independent draws).
//
// Scope / known limitation: this test exercises a *single* logical MX block (one
// 32-K block spanning two physical 16-wide K phases) — the core DIM=16 two-phase
// contribution. Cross-block accumulation at DIM=16 (K > 32, multiple logical
// blocks) additionally depends on draining the *last* weight-stationary compute
// out of the systolic array before the mvout; at DIM=16 (with the MX one-cycle
// output delay) the final block's compute is not fully clocked out by a trailing
// mvout, so its contribution is dropped. That WS last-compute drain interaction is
// orthogonal to the two-phase block-scaling novelty and is deferred (tracked with
// the K=96/128 split runs). See DOCS_MX/PLAN_MX.md (R3 / M5).
//
// Build prerequisite: compile against the DIM=16 MX params header
// (`gemmini_params_mxint8_dim16.h`, MX_ENABLED=1, DIM=16); with the stock header
// this test is a no-op. Single output tile (M = N = DIM = 16); MX v1 is WS-only
// and untransposed.

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

#if !MX_ENABLED || (DIM != 16)

int main() {
  printf("MX/DIM16 disabled in this build (need gemmini_params_mxint8_dim16.h); skipping\n");
  exit(0);
}

#else

#define M DIM
#define N DIM
#define K_MAX 64
#define KB_MAX (K_MAX / MX_BLOCK_SIZE)

static uint32_t lcg = 0x13572468u;
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

// Reset every buffer to a well-defined background: zero payloads and neutral
// scales (`encode(0)`) on all rows/cols, so no stale lane can corrupt a later
// case and the unread scale lanes stay valid (never 0xff).
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

// One MXINT8 GEMM producing the M x N output tile. The logical K dimension is
// `k` (a multiple of MX_BLOCK_SIZE=32); at DIM=16 each logical 32-K block is fed
// as TWO physical 16-wide K phases (preload(B half)/compute(A half)). The first
// half is buffered and its accumulator write is suppressed; the second half sums
// both halves' raw partials, scales once, and writes the int32 accumulator.
static void mxint8_hw_matmul(size_t k, size_t kb) {
  (void)k;
  const uint32_t acc_base = (1u << (ADDR_LEN - 1)) | (1u << (ADDR_LEN - 3)); // is_acc + read_full
  const uint32_t acc_accum = (1u << (ADDR_LEN - 2));                         // accumulate-on-write
  const uint32_t a_sp = 0;
  const size_t phases = 2 * kb;          // exactly two physical K phases per logical block
  const uint32_t b_sp = phases * DIM;    // B payload sits below A in the scratchpad

  gemmini_flush(0);
  gemmini_config_mxint8(true, 0);        // enable MX + reset the logical-K counter
  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);
  gemmini_config_st(DIM * sizeof(acc_t));
  // Scales: one E8M0 per logical 32-K block. A rows = M, B cols = N (= DIM here).
  gemmini_mvin_mxscale_a(&a_scale[0][0], 0, M, kb, KB_MAX);
  gemmini_mvin_mxscale_b(&b_scale[0][0], MX_SCALE_SP_ROWS / 2, kb, N, N);

  // Payloads: each physical phase is a DIM x DIM (16x16) slab. Phase p covers K
  // columns [p*DIM, p*DIM+DIM); p = 2*lb + h, so K offset p*DIM = lb*32 + h*16.
  gemmini_config_ld(K_MAX * sizeof(elem_t));
  for (size_t p = 0; p < phases; p++) {
    gemmini_mvin(&a_payload[0][p * DIM], a_sp + p * DIM);
  }
  gemmini_config_ld(N * sizeof(elem_t));
  for (size_t p = 0; p < phases; p++) {
    gemmini_mvin(&b_payload[p * DIM][0], b_sp + p * DIM);
  }

  // Two phases per logical block. The C address (and its accumulate bit) is keyed
  // by the logical block: block 0 overwrites, later blocks accumulate. Both halves
  // of a block share the same C address; the first half's write is suppressed in
  // hardware, so only the second half actually touches the accumulator.
  for (size_t p = 0; p < phases; p++) {
    const size_t lb = p / 2;
    const uint32_t c_addr = acc_base | (lb == 0 ? 0u : acc_accum);
    gemmini_preload(b_sp + p * DIM, c_addr);
    gemmini_compute_preloaded(a_sp + p * DIM, GARBAGE_ADDR);
  }

  gemmini_mvout(&c_hw[0][0], acc_base);
  gemmini_fence();
}

// Golden + hardware diff over the M x N tile for the data currently staged.
static int check(const char *label, size_t k, size_t kb) {
  const int rc = mxint8_ref_gemm_acc(&a_payload[0][0], &b_payload[0][0],
                                     &a_scale[0][0], &b_scale[0][0], &gold[0][0],
                                     M, N, k, K_MAX, N, N, KB_MAX, N);
  if (rc != 0) {
    printf("%-8s K=%d: golden rejected an invalid scale (rc=%d)\n", label, (int)k, rc);
    return 1;
  }

  mxint8_hw_matmul(k, kb);

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

// Deterministic probe over one logical block (K=32 = two 16-wide halves):
// all-ones payloads, eA[i]=6+(i%4), eB[j]=6+(j%3) ⇒ hw[i][j] = 32 << ((i%4)+(j%3)).
// Exact match ⇒ the two 16-wide raw partials were summed to 32 (not 16) and the
// per-block scale was applied once after the sum.
static int run_probe(void) {
  clear_all();
  const size_t k = MX_BLOCK_SIZE, kb = 1;
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < k; kk++) { a_payload[i][kk] = 1; }
    a_scale[i][0] = mxint8_e8m0_encode(6 + (int)(i % 4));
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = 1; }
  }
  for (size_t j = 0; j < N; j++) { b_scale[0][j] = mxint8_e8m0_encode(6 + (int)(j % 3)); }
  return check("probe", k, kb);
}

// Randomized fp32 packed through mxint8_pack.h, diffed against the golden. K is a
// multiple of MX_BLOCK_SIZE so the two-phase split is exact (no K-tail here).
static int run_random(size_t k) {
  clear_all();
  const size_t kb = k / MX_BLOCK_SIZE;
  for (size_t i = 0; i < M; i++) {
    const float row_mag = 1.0f + (float)(i % 5);
    for (size_t kk = 0; kk < k; kk++) { a_f[i][kk] = frand_signed(row_mag); }
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < N; j++) { b_f[kk][j] = frand_signed(1.0f + (float)(j % 3)); }
  }
  mxint8_pack_a(&a_f[0][0], M, k, K_MAX, KB_MAX, &a_payload[0][0], &a_scale[0][0]);
  mxint8_pack_b(&b_f[0][0], k, N, N, N, &b_payload[0][0], &b_scale[0][0]);
  return check("random", k, kb);
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int bad = 0;
  bad |= run_probe();      // one logical block, two phases, deterministic two-phase sum
  bad |= run_random(32);   // one logical block, two phases, randomized
  bad |= run_random(32);   // second independent random draw (lcg advanced)

  if (bad) {
    printf("mxint8_matmul_dim16: FAIL\n");
    exit(1);
  }
  printf("mxint8_matmul_dim16: PASS\n");
  exit(0);
}

#endif // MX_ENABLED && DIM == 16
