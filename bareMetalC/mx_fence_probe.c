// See LICENSE for license details.
//
// MXINT8 fence/pipelining fast-probe (plan P3, Step 0).
//
// Purpose: the cheapest possible signal for "do consecutive MX chunks pipeline?" The MX
// slowdown is a per-software-chunk serialized prologue (gemmini_fence + CONFIG_MXINT8 +
// scale-DMA); see DOCS_MX/PERF_ANALYSIS.md. This probe runs the SAME tiny per-chunk work
// at 1, 2, and 4 chunks and reports cycles + CounterFile events, so the per-chunk cost is
// read directly from the cycle ratio:
//   - fence present  (baseline): cycles(N chunks) ~= N * cycles(1 chunk); exe_active/cycles tiny
//   - chunks pipeline (P3 goal):  cycles(N chunks) <<  N * cycles(1 chunk); exe_active/cycles rises
//
// Chunking is forced purely in the I (row) dimension: N = DIM (one J-tile, Jp=1) and
// K = MX_BLOCK_SIZE (exactly one logical block, whole-block-safe at every DIM), so each
// invocation does i_chunk x 1 x 1 = i_chunk tiles of identical work and the chunk count is
// just M/(i_chunk*DIM). i_chunk mirrors tiled_matmul_mxint8's envelope (the smaller of the
// accumulator-half and A-scale-region bounds). Runs in seconds on a built sim; no rebuild
// for software-only steps. MX-only (stock has no chunked prologue to measure).

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#if MX_ENABLED
#include "include/mxint8_golden.h"
#endif

#if !MX_ENABLED
int main() {
  printf("MX disabled in this build (need an MX params header); skipping\n");
  exit(0);
}
#else

#define GOLDEN_ROWS 4

// D0: SINGLE-CHUNK shapes of increasing matmul count. All fit one tiler invocation
// (M<=2*DIM=i_chunk, N<=4*DIM=>Jp=4, K<=8 blocks=k_chunk at DIM32), so each is exactly ONE
// fenceless chunk = one CONFIG + one A/B scale-mvin pair + one LOOP_WS. Plotting cycles vs
// matmul count separates the PER-CHUNK fixed cost (intercept: setup + scale-load) from the
// PER-MATMUL cost (slope: compute + drain). matmuls = (M/DIM)*(N/DIM)*(K/DIM).
#define M_MAX (2 * DIM)
#define N_MAX (4 * DIM)
#define K_MAX (8 * MX_BLOCK_SIZE)
#define KB_MAX ((K_MAX + 31) / 32)

typedef struct { size_t m, n, k; } probe_shape_t;

static const probe_shape_t shapes[] = {
  {DIM,     DIM,     MX_BLOCK_SIZE},       // 1 matmul (minimal: pure per-chunk fixed cost)
  {DIM,     DIM,     4 * MX_BLOCK_SIZE},   // K-accumulate: +3 matmuls, +scale rows, same 1 tile
  {2 * DIM, 2 * DIM, MX_BLOCK_SIZE},       // 4 matmuls, more output tiles
  {2 * DIM, 4 * DIM, 4 * MX_BLOCK_SIZE},   // 32 matmuls
  {2 * DIM, 4 * DIM, 8 * MX_BLOCK_SIZE},   // 64 matmuls (= one 256^3-class chunk)
};

static elem_t a_payload[M_MAX][K_MAX] row_align(1);
static elem_t b_payload[K_MAX][N_MAX] row_align(1);
static acc_t c_hw[M_MAX][N_MAX] row_align(1);
static acc_t gold_row[N_MAX];
static mx_scale_t a_scale[M_MAX][KB_MAX] __attribute__((aligned(64)));
static mx_scale_t b_scale[KB_MAX][N_MAX] __attribute__((aligned(64)));

static uint32_t lcg = 0xa5a5a5a5u;
static uint32_t lcg_next(void) { lcg = lcg * 1664525u + 1013904223u; return lcg; }

static void gen_inputs(size_t m, size_t n, size_t k) {
  for (size_t i = 0; i < m; i++)
    for (size_t kk = 0; kk < k; kk++)
      a_payload[i][kk] = (elem_t)((int)(lcg_next() % 255) - 127);
  for (size_t kk = 0; kk < k; kk++)
    for (size_t j = 0; j < n; j++)
      b_payload[kk][j] = (elem_t)((int)(lcg_next() % 255) - 127);
  const size_t kb = (k + 31) / 32;
  for (size_t i = 0; i < m; i++)
    for (size_t b = 0; b < kb; b++)
      a_scale[i][b] = (mx_scale_t)(127 + (int)(lcg_next() % 5) - 2);
  for (size_t b = 0; b < kb; b++)
    for (size_t j = 0; j < n; j++)
      b_scale[b][j] = (mx_scale_t)(127 + (int)(lcg_next() % 5) - 2);
}

static void run_gemm(size_t m, size_t n, size_t k) {
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
  gemmini_extended_config_st(N_MAX * sizeof(acc_t), 0, ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ld(K_MAX * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(N_MAX * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);
  tiled_matmul_mxint8(m, n, k,
      &a_payload[0][0], &b_payload[0][0], /*D=*/NULL, &c_hw[0][0],
      &a_scale[0][0], &b_scale[0][0],
      K_MAX, N_MAX, 0, N_MAX, KB_MAX, N_MAX,
      /*full_C=*/true, /*low_D=*/false, /*act=*/0);
  gemmini_fence();
}

static int check_row(size_t row, size_t n, size_t k) {
  const int rc = mxint8_ref_gemm_acc(&a_payload[row][0], &b_payload[0][0],
                                     &a_scale[row][0], &b_scale[0][0], &gold_row[0],
                                     1, n, k, K_MAX, N_MAX, N_MAX, KB_MAX, N_MAX);
  if (rc != 0) { printf("golden rejected row %lu (rc=%d)\n", (unsigned long)row, rc); return 1; }
  for (size_t j = 0; j < n; j++)
    if (c_hw[row][j] != gold_row[j]) {
      printf("row %lu col %lu: hw=%d gold=%d\n",
             (unsigned long)row, (unsigned long)j, (int)c_hw[row][j], (int)gold_row[j]);
      return 1;
    }
  return 0;
}

static int run_shape(const probe_shape_t *s) {
  gen_inputs(s->m, s->n, s->k);
  gemmini_flush(0);
  counter_configure(0, EXE_ACTIVE_CYCLE);
  counter_configure(1, RESERVATION_STATION_ACTIVE_CYCLES);
  counter_configure(2, LOAD_ACTIVE_CYCLE);
  counter_configure(3, STORE_ACTIVE_CYCLE);
  counter_reset();
  const uint64_t start = read_cycles();
  run_gemm(s->m, s->n, s->k);
  const uint64_t end = read_cycles();
  gemmini_fence();
  const uint32_t c_active   = counter_read(0);
  const uint32_t c_rsactive = counter_read(1);
  const uint32_t c_ldactive = counter_read(2);
  const uint32_t c_stactive = counter_read(3);

  int bad = 0;
  for (int r = 0; r < GOLDEN_ROWS; r++) bad |= check_row(lcg_next() % s->m, s->n, s->k);

  const uint64_t cycles = end - start;
  const uint64_t matmuls = (s->m / DIM) * (s->n / DIM) * (s->k / DIM);
  printf("MXFENCE,dim=%d,matmuls=%lu,M=%lu,N=%lu,K=%lu,cycles=%llu,cyc_per_mm=%llu,"
         "exe_active=%u,rs_active=%u,ld_active=%u,st_active=%u,result=%s\n",
         DIM, (unsigned long)matmuls, (unsigned long)s->m, (unsigned long)s->n,
         (unsigned long)s->k, (unsigned long long)cycles,
         (unsigned long long)(cycles / (matmuls ? matmuls : 1)), c_active, c_rsactive,
         c_ldactive, c_stactive, bad ? "FAIL" : "PASS");
  return bad;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
#endif
  printf("MXFENCE probe (D0): DIM=%d -- single-chunk sweep; cycles vs matmuls separates "
         "per-chunk fixed cost (intercept) from per-matmul cost (slope)\n", DIM);
  int bad = 0;
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) bad |= run_shape(&shapes[s]);
  printf("mx_fence_probe: %s\n", bad ? "FAIL" : "PASS");
  exit(bad ? 1 : 0);
}

#endif // MX_ENABLED
