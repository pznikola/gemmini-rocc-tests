// See LICENSE for license details.
//
// First end-to-end MXINT8 hardware GEMM test (DIM=32, single output tile).
//
// Loads E8M0 scales into the sidecar MXScaleSRAM (`gemmini_mvin_mxscale_a/b`),
// runs the payload GEMM through an untransposed weight-stationary path with a
// full-width int32 accumulator output, and compares the hardware result against
// the bit-exact software golden `mxint8_ref_gemm_acc`.
//
// Two deterministic cases:
//   1. A single-block probe: all-ones payloads with per-row/per-column scales
//      chosen so hw[i][j] = raw << ((i%4)+(j%3)). This makes the block scaling
//      directly observable and pins down the per-output-row A-scale addressing.
//   2. A two-block probe: block0 contributes raw=32 and block1 contributes raw=64
//      under independently varied A/B scales, exercising cross-block accumulation.
//
// The expected output is computed analytically from the same scalar block-scaling
// helper as the software golden. This keeps the Verilator regression focused on
// hardware correctness instead of spending minutes in target-side fp32 packing and
// an O(M*N*K) reference loop.
//
// Build prerequisite: compile against the DIM=32 MX params header
// (`gemmini_params_mxint8_dim32.h`, MX_ENABLED=1); with the stock header this
// test is a no-op. Scope: single output tile (M = N = DIM); MX v1 is WS-only and
// untransposed, so the standard tiled_matmul path (which uses the transposer)
// cannot be used here.

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

static void clear_all(void) {
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < K_MAX; kk++) {
      a_payload[i][kk] = 0;
    }
    for (size_t b = 0; b < KB_MAX; b++) {
      a_scale[i][b] = mxint8_e8m0_encode(0);
    }
  }
  for (size_t kk = 0; kk < K_MAX; kk++) {
    for (size_t j = 0; j < N; j++) {
      b_payload[kk][j] = 0;
    }
  }
  for (size_t b = 0; b < KB_MAX; b++) {
    for (size_t j = 0; j < N; j++) {
      b_scale[b][j] = mxint8_e8m0_encode(0);
    }
  }
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      c_hw[i][j] = 0;
      gold[i][j] = 0;
    }
  }
}

static void build_expected(size_t kb) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      int64_t acc = 0;
      for (size_t b = 0; b < kb; b++) {
        // Payload block b uses A=(b+1), B=1 for all 32 lanes.
        const int64_t raw = (int64_t)MX_BLOCK_SIZE * (int64_t)(b + 1);
        acc += mxint8_scale_raw_block(raw,
            mxint8_e8m0_decode(a_scale[i][b]),
            mxint8_e8m0_decode(b_scale[b][j]));
      }
      gold[i][j] = mxint8_saturate_acc(acc);
    }
  }
}

// Run one MXINT8 GEMM on the accelerator: enable MX + reset the logical-K
// counter, load the A/B scale vectors into MXScaleSRAM, then preload each B block
// and compute each A block (untransposed WS), accumulating across the K blocks
// into a full-width int32 accumulator. Each WS compute is one logical MX block,
// so the K-lane counter advances one block per compute and the block scales are
// applied on the accumulator output.
static void mxint8_hw_matmul(size_t k, size_t kb) {
  const uint32_t acc_base = (1u << (ADDR_LEN - 1)) | (1u << (ADDR_LEN - 3)); // is_acc + read_full
  const uint32_t acc_accum = (1u << (ADDR_LEN - 2));                         // accumulate-on-write
  const uint32_t a_sp = 0;
  const uint32_t b_sp = KB_MAX * DIM;

  gemmini_flush(0);
  gemmini_config_mxint8(true, 0);
  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);
  gemmini_config_st(DIM * sizeof(acc_t));
  gemmini_mvin_mxscale_a(&a_scale[0][0], 0, M, kb, KB_MAX);
  gemmini_mvin_mxscale_b(&b_scale[0][0], MX_SCALE_SP_ROWS / 2, kb, N, N);

  gemmini_config_ld(K_MAX * sizeof(elem_t));
  for (size_t b = 0; b < kb; b++) {
    gemmini_mvin(&a_payload[0][b * MX_BLOCK_SIZE], a_sp + b * DIM);
  }
  gemmini_config_ld(N * sizeof(elem_t));
  for (size_t b = 0; b < kb; b++) {
    gemmini_mvin(&b_payload[b * MX_BLOCK_SIZE][0], b_sp + b * DIM);
  }

  for (size_t b = 0; b < kb; b++) {
    const uint32_t c_addr = acc_base | (b == 0 ? 0u : acc_accum);
    gemmini_preload(b_sp + b * DIM, c_addr);
    gemmini_compute_preloaded(a_sp + b * DIM, GARBAGE_ADDR);
  }

  gemmini_mvout(&c_hw[0][0], acc_base);
  gemmini_fence();
}

static int check(const char *label, size_t k, size_t kb) {
  printf("%s K=%d: hw begin\n", label, (int)k);
  mxint8_hw_matmul(k, kb);
  printf("%s K=%d: compare begin\n", label, (int)k);

  int mismatches = 0;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if (c_hw[i][j] != gold[i][j]) {
        mismatches++;
      }
    }
  }

  printf("%s K=%d: hw/gold row0[0..3]= %d/%d %d/%d %d/%d %d/%d  col0[0..3]= %d/%d %d/%d %d/%d %d/%d  %s (%d mism)\n",
         label, (int)k,
         (int)c_hw[0][0], (int)gold[0][0], (int)c_hw[0][1], (int)gold[0][1],
         (int)c_hw[0][2], (int)gold[0][2], (int)c_hw[0][3], (int)gold[0][3],
         (int)c_hw[0][0], (int)gold[0][0], (int)c_hw[1][0], (int)gold[1][0],
         (int)c_hw[2][0], (int)gold[2][0], (int)c_hw[3][0], (int)gold[3][0],
         mismatches ? "FAIL" : "PASS", mismatches);
  return mismatches ? 1 : 0;
}

// Deterministic probe: raw[i][j] = k (all-ones payloads), eA[i]=6+(i%4),
// eB[j]=6+(j%3) ⇒ shift = (i%4)+(j%3) ≥ 0, so hw[i][j] = k << shift. The per-row
// variation directly checks the per-output-row A-scale addressing.
static int run_probe(void) {
  const size_t k = MX_BLOCK_SIZE, kb = 1;
  clear_all();
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < k; kk++) {
      a_payload[i][kk] = 1;
    }
    a_scale[i][0] = mxint8_e8m0_encode(6 + (int)(i % 4));
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < N; j++) {
      b_payload[kk][j] = 1;
    }
  }
  for (size_t j = 0; j < N; j++) {
    b_scale[0][j] = mxint8_e8m0_encode(6 + (int)(j % 3));
  }
  build_expected(kb);
  return check("probe", k, kb);
}

static int run_two_block_probe(void) {
  const size_t k = 2 * MX_BLOCK_SIZE, kb = 2;
  clear_all();
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < k; kk++) {
      a_payload[i][kk] = kk < MX_BLOCK_SIZE ? 1 : 2;
    }
    a_scale[i][0] = mxint8_e8m0_encode(6 + (int)(i % 4));
    a_scale[i][1] = mxint8_e8m0_encode(4 + (int)(i % 3));
  }
  for (size_t kk = 0; kk < k; kk++) {
    for (size_t j = 0; j < N; j++) {
      b_payload[kk][j] = 1;
    }
  }
  for (size_t j = 0; j < N; j++) {
    b_scale[0][j] = mxint8_e8m0_encode(6 + (int)(j % 3));
    b_scale[1][j] = mxint8_e8m0_encode(5 + (int)(j % 2));
  }
  build_expected(kb);
  return check("two_blk", k, kb);
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int bad = 0;
  bad |= run_probe();      // single block, deterministic, per-row/col scales
  bad |= run_two_block_probe(); // two blocks, deterministic cross-block accumulate

  if (bad) {
    printf("mxint8_matmul_dim32: FAIL\n");
    exit(1);
  }
  printf("mxint8_matmul_dim32: PASS\n");
  exit(0);
}

#endif // MX_ENABLED
