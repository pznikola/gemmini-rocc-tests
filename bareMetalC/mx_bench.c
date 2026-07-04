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
// Correctness gates every data point: GOLDEN_POINTS sampled output cells are checked
// against the bit-exact reference (int32 GEMM for stock; MX block-scaled GEMM for MX).
// Machine-readable result lines:
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

#define GOLDEN_POINTS 16
#ifndef MX_BENCH_PRINT_GEOM
#define MX_BENCH_PRINT_GEOM 0
#endif
#ifndef MX_BENCH_COUNTER_SET
#define MX_BENCH_COUNTER_SET 0
#endif

static inline uint64_t read_insts(void) {
  uint64_t c;
  __asm__ volatile ("rdinstret %0" : "=r"(c));
  return c;
}

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

static elem_t a_payload[M_MAX][K_MAX] row_align(1) = {
  [0 ... M_MAX-1] = { [0 ... K_MAX-1] = 127 },
};
static elem_t b_payload[K_MAX][N_MAX] row_align(1) = {
  [0 ... K_MAX-1] = { [0 ... N_MAX-1] = 127 },
};
static acc_t c_hw[M_MAX][N_MAX] row_align(1);
#if MX_ENABLED
static mx_scale_t a_scale[M_MAX][KB_MAX] __attribute__((aligned(64))) = {
  [0 ... M_MAX-1] = { [0 ... KB_MAX-1] = 127 },
};
static mx_scale_t b_scale[KB_MAX][N_MAX] __attribute__((aligned(64))) = {
  [0 ... KB_MAX-1] = { [0 ... N_MAX-1] = 127 },
};
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

static size_t sample_index(size_t limit, uint32_t x) {
  return (size_t)(((uint64_t)(x & 0xffffu) * (uint64_t)limit) >> 16);
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

#if MX_ENABLED && MX_BENCH_PRINT_GEOM
static void print_mx_geom(size_t m, size_t n, size_t k) {
  const mxint8_geom_t g = mxint8_compute_geom(m, n, k);
  const size_t mx_IC = (g.i_tiles + g.i_chunk - 1) / g.i_chunk;
  const bool mx_b_reuse = (g.mx_JC * g.mx_KC <= 2);
  const bool mx_a_reuse = (g.mx_JC > 1);
  const bool k_chunked = (g.k_chunk < g.k_tiles);
  const bool mx_pairj_b_reuse = (DIM < MX_BLOCK_SIZE) && !k_chunked &&
      (g.mx_JC == 4) && (g.mx_KC == 1) &&
      (g.j_tiles == g.mx_JC * g.j_chunk) && (g.i_tiles == mx_IC * g.i_chunk);
  const size_t mx_pairj_width = 2;
  const bool mx_effective_b_reuse = mx_b_reuse || mx_pairj_b_reuse;
  const bool mx_bscale_residency_enough_i =
      (DIM < MX_BLOCK_SIZE) ? (mx_IC >= 2) : (mx_IC > 2);
  const bool mx_use_bscale_residency = mx_effective_b_reuse &&
      (g.mx_JC * g.mx_KC <= 4) && mx_bscale_residency_enough_i && !k_chunked;

  size_t chunks = 0;
  size_t fences = 0;
  size_t a_payload_loads = 0;
  size_t b_payload_loads = 0;
  size_t a_scale_loads = 0;
  size_t b_scale_loads = 0;
  bool mx_bscale_resident[4][2] = {{false}};
  bool first_chunk = true;
  size_t since_fence = 0;
  size_t prev_it = 0, prev_jt = 0, prev_kt = 0;

  const size_t total_chunks = mx_pairj_b_reuse
      ? (mx_IC * g.mx_JC * g.mx_KC)
      : ((g.i_tiles + g.i_chunk - 1) / g.i_chunk) *
        ((g.j_tiles + g.j_chunk - 1) / g.j_chunk) *
        ((g.k_tiles + g.k_chunk - 1) / g.k_chunk);

  for (size_t seq = 0; seq < total_chunks; seq++) {
    size_t mx_ic = 0, mx_jc = 0, mx_kc = 0, mx_pair_local_j = 0;
    bool mx_pair_group_start = false;

    if (mx_pairj_b_reuse) {
      const size_t chunks_per_pair_group = mx_IC * mx_pairj_width * g.mx_KC;
      const size_t pair_group = seq / chunks_per_pair_group;
      const size_t pair_rem = seq % chunks_per_pair_group;
      mx_ic = pair_rem / (mx_pairj_width * g.mx_KC);
      const size_t pair_inner = pair_rem % (mx_pairj_width * g.mx_KC);
      mx_pair_local_j = pair_inner / g.mx_KC;
      mx_kc = pair_inner % g.mx_KC;
      mx_jc = pair_group * mx_pairj_width + mx_pair_local_j;
      mx_pair_group_start = (mx_ic == 0 && mx_pair_local_j == 0 && mx_kc == 0);
    } else {
      mx_ic = seq / (g.mx_JC * g.mx_KC);
      const size_t rem = seq % (g.mx_JC * g.mx_KC);
      mx_jc = rem / g.mx_KC;
      mx_kc = rem % g.mx_KC;
    }

    const size_t i0 = mx_ic * g.i_chunk;
    const size_t j0 = mx_jc * g.j_chunk;
    const size_t k0 = mx_kc * g.k_chunk;
    const size_t it = (i0 + g.i_chunk <= g.i_tiles) ? g.i_chunk : (g.i_tiles - i0);
    const size_t jt = (j0 + g.j_chunk <= g.j_tiles) ? g.j_chunk : (g.j_tiles - j0);
    const size_t kt = (k0 + g.k_chunk <= g.k_tiles) ? g.k_chunk : (g.k_tiles - k0);
    const bool geom_same = (it == prev_it && jt == prev_jt && kt == prev_kt);
    const bool pair_group_barrier = mx_pairj_b_reuse && mx_pair_group_start && mx_jc != 0;
    const bool do_fence = k_chunked || first_chunk || !geom_same || pair_group_barrier;
    const int pp = do_fence ? 0 : (int)(since_fence % 2);
    const int pp_idx = (pp == 1) ? 1 : 0;
    const size_t mx_slot = mx_jc * g.mx_KC + mx_kc;
    const bool mx_a_reuse_hit = mx_a_reuse &&
        (mx_pairj_b_reuse ? (mx_pair_local_j >= 1) : (mx_jc >= 1));
    const bool mx_b_reuse_active = mx_b_reuse || mx_pairj_b_reuse;
    const bool load_a = !mx_a_reuse_hit;
    const bool load_b = !(mx_b_reuse_active && mx_ic >= 1);
    const bool bscale_hit = mx_use_bscale_residency &&
        (mx_slot < 4) && mx_bscale_resident[mx_slot][pp_idx];

    chunks++;
    if (do_fence) fences++;
    if (load_a) a_payload_loads++;
    if (load_b) b_payload_loads++;
    a_scale_loads++;
    if (!bscale_hit) b_scale_loads++;
    if (mx_use_bscale_residency && mx_slot < 4) {
      mx_bscale_resident[mx_slot][pp_idx] = true;
    }

    since_fence = do_fence ? 1 : (since_fence + 1);
    first_chunk = false;
    prev_it = it; prev_jt = jt; prev_kt = kt;
  }

  printf("MXGEOM,dim=%d,M=%lu,N=%lu,K=%lu,i_tiles=%lu,j_tiles=%lu,k_tiles=%lu,"
         "i_chunk=%lu,j_chunk=%lu,k_chunk=%lu,mx_IC=%lu,mx_JC=%lu,mx_KC=%lu,"
         "chunks=%lu,fences=%lu,a_reuse=%u,b_reuse=%u,a_payload_loads=%lu,"
         "b_payload_loads=%lu,a_scale_loads=%lu,b_scale_loads=%lu,k_chunked=%u,"
         "pairj_reuse=%u,bscale_residency=%u\n",
         DIM, (unsigned long)m, (unsigned long)n, (unsigned long)k,
         (unsigned long)g.i_tiles, (unsigned long)g.j_tiles, (unsigned long)g.k_tiles,
         (unsigned long)g.i_chunk, (unsigned long)g.j_chunk, (unsigned long)g.k_chunk,
         (unsigned long)mx_IC, (unsigned long)g.mx_JC, (unsigned long)g.mx_KC,
         (unsigned long)chunks, (unsigned long)fences,
         mx_a_reuse ? 1u : 0u, mx_b_reuse ? 1u : 0u,
         (unsigned long)a_payload_loads, (unsigned long)b_payload_loads,
         (unsigned long)a_scale_loads, (unsigned long)b_scale_loads,
         k_chunked ? 1u : 0u, mx_pairj_b_reuse ? 1u : 0u,
         mx_use_bscale_residency ? 1u : 0u);
}
#endif

// Bit-exact reference for one sampled output cell.
static int reference_cell(size_t row, size_t col, size_t k, acc_t *gold) {
#if MX_ENABLED
  const size_t k_blocks = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
  int64_t acc = 0;

  for (size_t block = 0; block < k_blocks; block++) {
    const mx_scale_t scale_a = a_scale[row][block];
    const mx_scale_t scale_b = b_scale[block][col];

    if (!mxint8_e8m0_is_valid(scale_a) || !mxint8_e8m0_is_valid(scale_b))
      return -1;

    int64_t raw = 0;
    for (size_t t = 0; t < MX_BLOCK_SIZE; t++) {
      const size_t kk = block * MX_BLOCK_SIZE + t;
      if (kk < k)
        raw += (int32_t)a_payload[row][kk] * (int32_t)b_payload[kk][col];
    }

    acc += mxint8_scale_raw_block(raw, mxint8_e8m0_decode(scale_a),
                                  mxint8_e8m0_decode(scale_b));
  }
  *gold = mxint8_saturate_acc(acc);
#else
  acc_t sum = 0;
  for (size_t kk = 0; kk < k; kk++) {
    sum += (acc_t)a_payload[row][kk] * (acc_t)b_payload[kk][col];
  }
  *gold = sum;
#endif
  return 0;
}

static int check_cell(size_t row, size_t col, size_t k) {
  acc_t gold = 0;
  const int rc = reference_cell(row, col, k, &gold);
  if (rc != 0) {
    printf("golden rejected scale at row %lu col %lu (rc=%d)\n",
           (unsigned long)row, (unsigned long)col, rc);
    return 1;
  }
  if (c_hw[row][col] != gold) {
    printf("row %lu col %lu: hw=%d gold=%d\n",
           (unsigned long)row, (unsigned long)col, (int)c_hw[row][col], (int)gold);
    return 1;
  }
  return 0;
}

static int run_shape(const bench_shape_t *s) {
  printf("TRACE,enter_run_shape,M=%lu,N=%lu,K=%lu\n",
         (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k);
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
  // Step-1/3 feed-bubble partition (Phase C regime): a loop is *configured* ~87% of the timed
  // region (loopmm_active) yet the mesh is fed only ~49% (matmul_in_progress) at 256³ — so on
  // top of the host-issue gap there is an INTRA-loop mesh-feed bubble. These 8 slots localize
  // it. spadA_wait/spadB_wait = the controller wants to feed A/B but the scratchpad read resp
  // isn't valid (the pre-Phase-C dominant stall — re-measure post-repack). overlap_haz = the
  // mesh can't start the next matmul until the previous output drained (drain serialization).
  // preload_haz = preload RAW. no_cmd = EX starved (now reset *after* the untimed pretile, so
  // it no longer counts the host pretile window). matmul_in_progress / loopmm_active = mesh-busy
  // and unroller-configured references. All events already wired in the RTL (no sim rebuild).
#if MX_BENCH_COUNTER_SET == 1
  counter_configure(0, MX_DBG_NO_CMD_CYCLE);                // EX controller starved
  counter_configure(1, MX_EX_POOL_EMPTY_CYCLE);             // no EX entry available
  counter_configure(2, MX_DBG_EX_READY_CYCLE);              // EX ready but not issued
  counter_configure(3, MX_DBG_EX_BLOCKED_CYCLE);            // EX dependency-blocked
  counter_configure(4, MX_DBG_EX_INFLIGHT_CYCLE);           // EX issued but incomplete
  counter_configure(5, MX_DBG_EX_POOL_FULL_CYCLE);          // EX pool throttles unroller
  counter_configure(6, MX_DBG_LD_INFLIGHT_CYCLE);           // loads issued but incomplete
  counter_configure(7, LOOP_MATMUL_ACTIVE_CYCLES);          // unroller configured (reference)
#elif MX_BENCH_COUNTER_SET == 2
  counter_configure(0, MX_EX_BLOCKED_ON_SCALE_CYCLE);        // EX blocked by any live scale mvin
  counter_configure(1, MX_EX_BLOCKED_SCALE_ONLY_CYCLE);      // scale dep is the only EX blocker
  counter_configure(2, MX_EX_BLOCKED_NONSCALE_CYCLE);        // EX blocked by non-scale deps
  counter_configure(3, MX_DBG_LD_POOL_FULL_CYCLE);           // LD pool full
  counter_configure(4, MX_DBG_LD_BLOCKED_CYCLE);             // LD dependency-blocked
  counter_configure(5, MX_DBG_ST_POOL_FULL_CYCLE);           // ST pool full
  counter_configure(6, MX_DBG_ST_INFLIGHT_CYCLE);            // ST issued but incomplete
  counter_configure(7, MX_SCALE_DMA_ACTIVE_CYCLE);           // scale-load controller busy
#else
  counter_configure(0, MX_DBG_MATMUL_IN_PROGRESS_CYCLE);    // mesh busy (reference)
  counter_configure(1, LOOP_MATMUL_ACTIVE_CYCLES);          // unroller configured (reference)
  counter_configure(2, SCRATCHPAD_A_WAIT_CYCLE);            // A feed: ctrl wants A, spad not ready
  counter_configure(3, SCRATCHPAD_B_WAIT_CYCLE);            // B feed: ctrl wants B, spad not ready
  counter_configure(4, EXE_OVERLAP_HAZ_CYCLE);              // next matmul blocked on prior drain
  counter_configure(5, EXE_PRELOAD_HAZ_CYCLE);              // preload RAW hazard
  counter_configure(6, MX_DBG_NO_CMD_CYCLE);                // EX controller starved
  counter_configure(7, RDMA_ACTIVE_CYCLE);                  // read-DMA engine active (reference)
#endif
#if MX_ENABLED && MX_BENCH_CPU_TIMING
  g_mx_repack_cyc = 0; g_mx_issue_cyc = 0;  // Phase-0c CPU-cost localization
  g_mx_fence_cyc = 0; g_mx_config_cyc = 0; g_mx_scalea_cyc = 0; g_mx_scaleb_cyc = 0;
#endif
#if MX_ENABLED
  // Phase C: pre-tile the constant B-scales ONCE here (untimed). This calls the repack
  // directly (not the timed wrapper), so g_mx_repack_cyc stays 0; the timed GEMM below reads
  // b_pretiled and never repacks. Bytes are identical to the in-loop repack -> bit-exact.
  mxint8_pretile_b_scales(s->m, s->n, s->k, &b_scale[0][0], N_MAX, b_pretiled);
#if MX_BENCH_PRINT_GEOM
  print_mx_geom(s->m, s->n, s->k);
#endif
#endif
  printf("TRACE,pre_run_gemm,M=%lu\n", (unsigned long)s->m);
  // Reset AFTER the untimed pretile so every counter aligns to the timed GEMM window.
  counter_reset();
#if MX_BENCH_COUNTER_SET == 3
  const uint64_t inst_start = read_insts();
#endif
  const uint64_t start = read_cycles();
  run_gemm(s->m, s->n, s->k);
  const uint64_t end = read_cycles();
#if MX_BENCH_COUNTER_SET == 3
  const uint64_t inst_end = read_insts();
#endif
  printf("TRACE,post_run_gemm,M=%lu\n", (unsigned long)s->m);
  // run_gemm() already includes the completion fence that defines the timed window.
  printf("TRACE,pre_counter_read,M=%lu\n", (unsigned long)s->m);
  const uint32_t c0 = counter_read(0);
  const uint32_t c1 = counter_read(1);
  const uint32_t c2 = counter_read(2);
  const uint32_t c3 = counter_read(3);
  const uint32_t c4 = counter_read(4);
  const uint32_t c5 = counter_read(5);
  const uint32_t c6 = counter_read(6);
  const uint32_t c7 = counter_read(7);
  printf("TRACE,post_counter_read,M=%lu\n", (unsigned long)s->m);

  int bad = 0;
  for (int p = 0; p < GOLDEN_POINTS; p++) {
    const size_t row = sample_index(s->m, lcg_next());
    const size_t col = sample_index(s->n, lcg_next());
    bad |= check_cell(row, col, s->k);
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
#if MX_BENCH_COUNTER_SET == 1
  printf("MXPART,impl=%s,dim=%d,M=%lu,N=%lu,K=%lu,cycles=%llu,no_cmd=%u,"
         "ex_pool_empty=%u,ex_ready=%u,ex_blocked=%u,ex_inflight=%u,ex_pool_full=%u,"
         "ld_inflight=%u,loopmm_active=%u\n",
#if MX_ENABLED
         "mx",
#else
         "stock",
#endif
         DIM, (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k,
         (unsigned long long)cycles, c0, c1, c2, c3, c4, c5, c6, c7);
#elif MX_BENCH_COUNTER_SET == 2
  printf("MXSCALEDEP,impl=%s,dim=%d,M=%lu,N=%lu,K=%lu,cycles=%llu,"
         "ex_blocked_on_scale=%u,ex_blocked_scale_only=%u,ex_blocked_nonscale=%u,"
         "ld_pool_full=%u,ld_blocked=%u,st_pool_full=%u,st_inflight=%u,scale_dma_active=%u\n",
#if MX_ENABLED
         "mx",
#else
         "stock",
#endif
         DIM, (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k,
         (unsigned long long)cycles, c0, c1, c2, c3, c4, c5, c6, c7);
#elif MX_BENCH_COUNTER_SET == 3
  const uint64_t instret = inst_end - inst_start;
  const uint64_t cyc_per_inst_x1000 = instret ? (cycles * 1000ULL) / instret : 0;
  printf("MXINST,impl=%s,dim=%d,M=%lu,N=%lu,K=%lu,cycles=%llu,"
         "instret=%llu,cyc_per_inst_x1000=%llu,matmul_in_progress=%u,"
         "loopmm_active=%u,no_cmd=%u,rdma_active=%u\n",
#if MX_ENABLED
         "mx",
#else
         "stock",
#endif
         DIM, (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k,
         (unsigned long long)cycles, (unsigned long long)instret,
         (unsigned long long)cyc_per_inst_x1000, c0, c1, c6, c7);
#else
  printf("MXCOUNT,impl=%s,dim=%d,M=%lu,N=%lu,K=%lu,cycles=%llu,matmul_in_progress=%u,"
         "loopmm_active=%u,spadA_wait=%u,spadB_wait=%u,overlap_haz=%u,preload_haz=%u,"
         "no_cmd=%u,rdma_active=%u\n",
#if MX_ENABLED
         "mx",
#else
         "stock",
#endif
         DIM, (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k,
         (unsigned long long)cycles, c0, c1, c2, c3, c4, c5, c6, c7);
#endif
#if MX_ENABLED && MX_BENCH_CPU_TIMING
  printf("MXCPU,impl=mx,dim=%d,M=%lu,N=%lu,K=%lu,cycles=%llu,repack_cyc=%llu,issue_cyc=%llu,"
         "fence_cyc=%llu,config_cyc=%llu,scalea_cyc=%llu,scaleb_cyc=%llu\n",
         DIM, (unsigned long)s->m, (unsigned long)s->n, (unsigned long)s->k,
         (unsigned long long)cycles, (unsigned long long)g_mx_repack_cyc,
         (unsigned long long)g_mx_issue_cyc, (unsigned long long)g_mx_fence_cyc,
         (unsigned long long)g_mx_config_cyc, (unsigned long long)g_mx_scalea_cyc,
         (unsigned long long)g_mx_scaleb_cyc);
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
  printf("TRACE,inputs_preloaded\n");

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
