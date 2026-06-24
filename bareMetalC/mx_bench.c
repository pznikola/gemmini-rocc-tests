// See LICENSE for license details.
//
// Stock-vs-MXINT8 benchmark (plan Stage C / P6 subset). ONE source, built per config via
// the header-staging pattern: with a stock params header (MX_ENABLED=0) the GEMM runs
// through `tiled_matmul_auto` (WS); with an MX header it runs through
// `tiled_matmul_mxint8`. Same shapes, same int8 payload data, same DIM per pair, so the
// MAC count is identical and the cycle delta is attributable to the MX metadata path.
//
// Fairness boundary: inputs are generated host-side at init (integer payloads + E8M0
// scales directly — quantization/packing is excluded on both sides; stock int8 needs
// none). The timed region is the tiled_matmul call + fence: payload mvins (+ scale
// mvins and per-invocation fences on the MX side), compute, and mvout. full_C=true on
// both sides (identical 4-byte/element mvout traffic).
//
// Correctness gates every data point: GOLDEN_ROWS sampled output rows are checked
// against the bit-exact reference (int32 GEMM for stock; `mxint8_ref_gemm_acc` rows for
// MX). Machine-readable result lines:
//   MXBENCH,impl=<stock|mx>,dim=<D>,M=,N=,K=,cycles=,macs=,ideal_cycles=,util_pct=,result=<PASS|FAIL>
//
// Shape table is gated by DIM to keep Verilator runtimes bounded (see plan Stage C).

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

#define GOLDEN_ROWS 8

// Shape limits (bytes are dominated by B and C for the BERT shapes).
#if DIM >= 8
// Arrays are sized for the largest shape actually run (below). Large static footprints
// with large leading dimensions stall stock tiled_matmul_auto on this Verilator harness,
// so the shape suite is a square sweep with matching (contiguous) array strides.
#define M_MAX 256
#define N_MAX 256
#define K_MAX 256
#else
#define M_MAX 256
#define N_MAX 256
#define K_MAX 256
#endif
#define KB_MAX ((K_MAX + 31) / 32)

typedef struct {
  size_t m, n, k;
} bench_shape_t;

static const bench_shape_t shapes[] = {
  {64, 64, 64},
  {128, 128, 128},
  {256, 256, 256},
};

static elem_t a_payload[M_MAX][K_MAX] row_align(1);
static elem_t b_payload[K_MAX][N_MAX] row_align(1);
static acc_t c_hw[M_MAX][N_MAX] row_align(1);
static acc_t gold_row[N_MAX];
#if MX_ENABLED
static mx_scale_t a_scale[M_MAX][KB_MAX] __attribute__((aligned(64)));
static mx_scale_t b_scale[KB_MAX][N_MAX] __attribute__((aligned(64)));
#endif

static uint32_t lcg = 0xa5a5a5a5u;
static uint32_t lcg_next(void) {
  lcg = lcg * 1664525u + 1013904223u;
  return lcg;
}

// Integer payloads in the packer-reachable range [-127, 127] and small-exponent E8M0
// scales (so MX results stay far from the int32 saturation envelope).
static void gen_inputs(size_t m, size_t n, size_t k) {
  for (size_t i = 0; i < m; i++) {
    for (size_t kk = 0; kk < k; kk++) {
      a_payload[i][kk] = (elem_t)((int)(lcg_next() % 255) - 127);
    }
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < n; j++) {
      b_payload[kk][j] = (elem_t)((int)(lcg_next() % 255) - 127);
    }
  }
#if MX_ENABLED
  const size_t kb = (k + 31) / 32;
  for (size_t i = 0; i < m; i++) {
    for (size_t b = 0; b < kb; b++) {
      a_scale[i][b] = (mx_scale_t)(127 + (int)(lcg_next() % 5) - 2);  // exp in [-2, 2]
    }
  }
  for (size_t b = 0; b < kb; b++) {
    for (size_t j = 0; j < n; j++) {
      b_scale[b][j] = (mx_scale_t)(127 + (int)(lcg_next() % 5) - 2);
    }
  }
#endif
}

static void run_gemm(size_t m, size_t n, size_t k) {
#if MX_ENABLED
  // The MX loop unroller (gemmini_loop_ws_mx) does NOT self-configure the accelerator, so
  // ex/st/ld must be set up first (the stock tiled_matmul_auto below self-configures, which
  // is why only the MX path needs this). Without it the scratchpad DMA runs with stale
  // config and the spad TL slave rejects the load (TLMonitor "Get type not supported").
  gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
  gemmini_extended_config_st(N_MAX * sizeof(acc_t), 0, ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ld(K_MAX * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0); // A
  gemmini_extended3_config_ld(N_MAX * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1); // B
  gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);                      // D (none)
  tiled_matmul_mxint8(m, n, k,
      &a_payload[0][0], &b_payload[0][0], /*D=*/NULL, &c_hw[0][0],
      &a_scale[0][0], &b_scale[0][0],
      K_MAX, N_MAX, 0, N_MAX, KB_MAX, N_MAX,
      /*full_C=*/true, /*low_D=*/false, /*act=*/0);
#else
  tiled_matmul_auto(m, n, k,
      &a_payload[0][0], &b_payload[0][0], /*D=*/NULL, &c_hw[0][0],
      K_MAX, N_MAX, 0, N_MAX,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      /*act=*/NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, /*repeating_bias=*/false,
      /*transpose_A=*/false, /*transpose_B=*/false,
      /*full_C=*/true, /*low_D=*/false,
      /*weightA=*/3, WS);
#endif
  gemmini_fence();
}

// Bit-exact reference for one sampled output row.
static int check_row(size_t row, size_t n, size_t k) {
#if MX_ENABLED
  const int rc = mxint8_ref_gemm_acc(&a_payload[row][0], &b_payload[0][0],
                                     &a_scale[row][0], &b_scale[0][0], &gold_row[0],
                                     1, n, k, K_MAX, N_MAX, N_MAX, KB_MAX, N_MAX);
  if (rc != 0) {
    printf("golden rejected scale on row %lu (rc=%d)\n", (unsigned long)row, rc);
    return 1;
  }
#else
  for (size_t j = 0; j < n; j++) {
    acc_t sum = 0;
    for (size_t kk = 0; kk < k; kk++) {
      sum += (acc_t)a_payload[row][kk] * (acc_t)b_payload[kk][j];
    }
    gold_row[j] = sum;
  }
#endif
  for (size_t j = 0; j < n; j++) {
    if (c_hw[row][j] != gold_row[j]) {
      printf("row %lu col %lu: hw=%d gold=%d\n",
             (unsigned long)row, (unsigned long)j, (int)c_hw[row][j], (int)gold_row[j]);
      return 1;
    }
  }
  return 0;
}

static int run_shape(const bench_shape_t *s) {
  gen_inputs(s->m, s->n, s->k);

  gemmini_flush(0);
  // Hazard/attribution counters (P-A.2): localize where the execute pipeline spends
  // cycles. Configured outside the timed region so MXBENCH cycle counts are unaffected.
  // D1 diagnostic set (no RTL rebuild): localize the ~127k idle. RDMA/WDMA_ACTIVE = the
  // actual DMA engines (capture the scale-load too if it shares the read DMA); DMA_TLB_MISS
  // = TLB thrashing (the per-row scale DMA is a prime suspect); SCRATCHPAD_B_WAIT = mesh feed
  // starved; LOOP_MATMUL_ACTIVE = unroller busy; RS_FULL = back-pressure.
  // B-feed mechanism probe: spadB_wait dominates (85% in MX vs 37% stock). Discriminate WHY
  // the B source can't deliver. overlap_haz = mesh can't start the next matmul until the
  // previous output has drained (drain serialization, the prime suspect); preload_haz =
  // preload hazard; spadA_wait vs spadB_wait = is the starvation B-specific or general feed;
  // ctrlq_block = the mesh control queue is blocked; rs_active = RS occupancy.
  counter_configure(0, EXE_ACTIVE_CYCLE);            // compute state
  counter_configure(1, MX_DBG_WAIT_CMD_CYCLE);       // waiting_for_cmd (RS-starved)
  counter_configure(2, MX_DBG_ENQ_NOT_READY_CYCLE);  // ctrl-q can't accept (gates read issue)
  counter_configure(3, MX_DBG_REQ_STALL_CYCLE);      // matmul entry blocked (mesh.req.ready low)
  counter_configure(4, MX_DBG_DRAINING_CYCLE);       // mesh emitting committed output (drain)
  counter_configure(5, LOOP_MATMUL_ACTIVE_CYCLES);
  counter_configure(6, RESERVATION_STATION_FULL_CYCLES);
  counter_configure(7, SCRATCHPAD_B_WAIT_CYCLE);
  counter_reset();
  const uint64_t start = read_cycles();
  run_gemm(s->m, s->n, s->k);
  const uint64_t end = read_cycles();
  gemmini_fence();
  const uint32_t c_active   = counter_read(0);  // EXE_ACTIVE (compute)
  const uint32_t c_rdbytes  = counter_read(1);  // WAIT_CMD
  const uint32_t c_wrbytes  = counter_read(2);  // ENQ_NOT_READY
  const uint32_t c_ldactive = counter_read(3);  // REQ_STALL
  const uint32_t c_stactive = counter_read(4);  // DRAINING
  const uint32_t c_ldwait   = counter_read(5);  // LOOP_MATMUL_ACTIVE
  const uint32_t c_stwait   = counter_read(6);  // RS_FULL
  const uint32_t c_rsactive = counter_read(7);  // SCRATCHPAD_B_WAIT

  int bad = 0;
  for (int r = 0; r < GOLDEN_ROWS; r++) {
    bad |= check_row(lcg_next() % s->m, s->n, s->k);
  }

  const uint64_t cycles = end - start;
  const uint64_t macs = (uint64_t)s->m * s->n * s->k;
  const uint64_t ideal = macs / ((uint64_t)DIM * DIM);
  const uint64_t util_pct = cycles ? (100 * ideal) / cycles : 0;

  printf("MXBENCH,impl=%s,dim=%d,M=%lu,N=%lu,K=%lu,cycles=%llu,macs=%llu,"
         "ideal_cycles=%llu,util_pct=%llu,result=%s\n",
#if MX_ENABLED
         "mx",
#else
         "stock",
#endif
         DIM, (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k,
         (unsigned long long)cycles, (unsigned long long)macs,
         (unsigned long long)ideal, (unsigned long long)util_pct,
         bad ? "FAIL" : "PASS");
  printf("MXCOUNT,impl=%s,dim=%d,M=%lu,N=%lu,K=%lu,cycles=%llu,exe_active=%u,"
         "wait_cmd=%u,enq_not_ready=%u,req_stall=%u,draining=%u,loopmm_active=%u,"
         "rs_full=%u,spadB_wait=%u\n",
#if MX_ENABLED
         "mx",
#else
         "stock",
#endif
         DIM, (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k,
         (unsigned long long)cycles, c_active, c_rdbytes, c_wrbytes, c_ldactive,
         c_stactive, c_ldwait, c_stwait, c_rsactive);
  return bad;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int bad = 0;
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
    bad |= run_shape(&shapes[s]);
  }

  if (bad) {
    printf("mx_bench: FAIL\n");
    exit(1);
  }
  printf("mx_bench: PASS\n");
  exit(0);
}
