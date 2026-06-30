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
// Known-good array sizing (matches the original suite). Tight 64-strides deadlocked the MX
// gemm (separate non-square/KB issue), so keep the original 256 strides for the trace.
#define M_MAX 256
#define N_MAX 256
#define K_MAX 256
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
// Phase C: offline-tiled B-scale image (weights are constant). Filled once per shape outside
// the timed region; the timed GEMM then reads it and does zero repack. 4 slots is the cache
// envelope (mx_JC*mx_KC <= 4 for these shapes).
static mx_scale_t b_pretiled[4 * MX_PRETILE_SLOT_ELEMS] __attribute__((aligned(64)));
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
  tiled_matmul_mxint8_pretiled(m, n, k,
      &a_payload[0][0], &b_payload[0][0], /*D=*/NULL, &c_hw[0][0],
      &a_scale[0][0], &b_scale[0][0],
      K_MAX, N_MAX, 0, N_MAX, KB_MAX, N_MAX,
      /*full_C=*/true, /*low_D=*/false, /*act=*/0,
      /*b_pretiled=*/b_pretiled);
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
  printf("TRACE,enter_run_shape,M=%lu,N=%lu,K=%lu\n",
         (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k);
  gen_inputs(s->m, s->n, s->k);
  printf("TRACE,gen_inputs_done,M=%lu\n", (unsigned long)s->m);

  gemmini_flush(0);
  printf("TRACE,flush_done,M=%lu\n", (unsigned long)s->m);
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
  // Phase-0 gate partition: split the 86% waiting_for_cmd. no_cmd = RS not issuing (upstream
  // dependency); cmd_blocked = EX has a command but a config/hazard barrier holds it;
  // matmul_in_progress / hold_not_drain = mesh tag occupancy (computed-but-not-draining).
  // RS-internal partition: when the EX controller has no command (no_cmd), why? ex_ready =
  // RS has a ready EX entry it isn't issuing (issue-handshake gate); ex_blocked = unissued
  // EX entries dependency-blocked (dependency gate); ex_inflight = EX entries issued but not
  // completed (completion-latency gate); ld_inflight = loads outstanding (ld_ahead gate).
  // rs_full re-included in the SAME run to settle CPU-vs-hardware with no cross-probe.
  // Phase-0b DAE-inversion partition: split the ~86% no_cmd (EX controller starved) into its
  // true cause. blk_scale = an unissued EX entry is dep-blocked with a live scale-mvin dep
  // (the decoupled-access-execute inversion: matmul issue gated on scale-DMA *completion*);
  // blk_scale_only = the scale dep is its ONLY blocker (relaxing it would free the matmul);
  // blk_nonscale = blocked by some non-scale dep (a different gate); ex_pool_empty = no EX
  // entry at all (unroller not delivering: idle or ld_ahead/upstream stall). ex_ready = RS
  // has a ready EX it isn't issuing (consume/FSM gate); ex_inflight = issued-not-completed
  // (completion latency); ld_blocked = loads dep-blocked. Decision: if blk_scale ~= no_cmd,
  // the scale-mvin completion gate is confirmed => Phase 1A (relax the matmul->scale dep).
  // Phase-0b split of the dominant ex_pool_empty (~82%): is the unroller IDLE (no loop to
  // run => command/loop-level starvation, Phase 1C) or BUSY-but-not-delivering (loopmm_active
  // high while the pool is empty => ld_ahead/feed stall inside a chunk, Phase 1B)? loopmm_active
  // and rs_active are stock counters already in the RTL, so this needs no sim rebuild. Also
  // observe mesh tag occupancy (matmul_in_progress), the config barrier (cmd_blocked), and the
  // residual non-scale EX block.
  // Phase-0c DMA localization of the ~77% inter-chunk idle: is the mesh starved because the
  // next chunk's payloads load serially (fence -> cold DMA, no overlap with prior compute)?
  // load_active/load_dma_wait = payload load controller; rdma_active = read DMA engine; tlb_miss
  // = TLB thrash; scale_dma = scale-mvin DMA. Compared against mesh-busy (matmul_in_progress)
  // and unroller-configured (loopmm). All are stock counters (valid in both sims), no rebuild.
  counter_configure(0, MX_DBG_NO_CMD_CYCLE);                // EX controller starved (denominator)
  counter_configure(1, MX_DBG_MATMUL_IN_PROGRESS_CYCLE);    // mesh busy (reference)
  counter_configure(2, LOAD_ACTIVE_CYCLE);                  // payload load DMA active
  counter_configure(3, LOAD_DMA_WAIT_CYCLE);                // load controller waiting on DMA
  counter_configure(4, RDMA_ACTIVE_CYCLE);                  // read-DMA engine active
  counter_configure(5, DMA_TLB_MISS_CYCLE);                 // TLB miss stalls
  counter_configure(6, MX_SCALE_DMA_ACTIVE_CYCLE);          // scale-mvin DMA active
  counter_configure(7, LOOP_MATMUL_ACTIVE_CYCLES);          // unroller configured (reference)
  counter_reset();
#if MX_ENABLED
  g_mx_repack_cyc = 0; g_mx_issue_cyc = 0;  // Phase-0c CPU-cost localization
  // Phase C: pre-tile the constant B-scales ONCE here (untimed). This calls the repack
  // directly (not the timed wrapper), so g_mx_repack_cyc stays 0; the timed GEMM below reads
  // b_pretiled and never repacks. Bytes are identical to the in-loop repack -> bit-exact.
  mxint8_pretile_b_scales(s->m, s->n, s->k, &b_scale[0][0], N_MAX, b_pretiled);
#endif
  printf("TRACE,pre_run_gemm,M=%lu\n", (unsigned long)s->m);
  const uint64_t start = read_cycles();
  run_gemm(s->m, s->n, s->k);
  const uint64_t end = read_cycles();
  printf("TRACE,post_run_gemm,M=%lu\n", (unsigned long)s->m);
  gemmini_fence();
  const uint32_t c_no_cmd     = counter_read(0);  // NO_CMD
  const uint32_t c_mm_prog    = counter_read(1);  // MATMUL_IN_PROGRESS
  const uint32_t c_ld_active  = counter_read(2);  // LOAD_ACTIVE
  const uint32_t c_ld_wait    = counter_read(3);  // LOAD_DMA_WAIT
  const uint32_t c_rdma       = counter_read(4);  // RDMA_ACTIVE
  const uint32_t c_tlb_miss   = counter_read(5);  // DMA_TLB_MISS
  const uint32_t c_scale_dma  = counter_read(6);  // MX_SCALE_DMA_ACTIVE
  const uint32_t c_loopmm     = counter_read(7);  // LOOP_MATMUL_ACTIVE

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
  printf("MXCOUNT,impl=%s,dim=%d,M=%lu,N=%lu,K=%lu,cycles=%llu,no_cmd=%u,"
         "matmul_in_progress=%u,load_active=%u,load_dma_wait=%u,rdma_active=%u,"
         "tlb_miss=%u,scale_dma=%u,loopmm_active=%u\n",
#if MX_ENABLED
         "mx",
#else
         "stock",
#endif
         DIM, (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k,
         (unsigned long long)cycles, c_no_cmd, c_mm_prog, c_ld_active, c_ld_wait,
         c_rdma, c_tlb_miss, c_scale_dma, c_loopmm);
#if MX_ENABLED
  printf("MXCPU,impl=mx,dim=%d,M=%lu,N=%lu,K=%lu,cycles=%llu,repack_cyc=%llu,issue_cyc=%llu\n",
         DIM, (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k,
         (unsigned long long)cycles, (unsigned long long)g_mx_repack_cyc,
         (unsigned long long)g_mx_issue_cyc);
#endif
  return bad;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  printf("TRACE,enter_main\n");

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
