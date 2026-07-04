// See LICENSE for license details.
//
// MXINT8 DIM=32 corner cases (single full output tile, M = N = DIM).
//
// Reuses the untransposed weight-stationary manual sequence from
// mxint8_matmul_dim32.c, with full-width int32 accumulator output, and diffs the
// hardware against the bit-exact golden `mxint8_ref_gemm_acc`. All inputs are set
// up directly as integers (no fp packer), so the cases are cheap on the core.
//
// Cases: all-zero payloads; max +127 and -127 payloads with shifts that saturate
// the int32 accumulator (int32-range and 64-bit-overflow regimes — exercises the
// R2-2 scalePowerOfTwo/saturate path); -128 payloads (OCP spec 5.3.4 leaves -128
// unused for *encoders* — our packer never emits it — but conformant third-party
// data may contain it; checked exact, int32-saturating, and negative-shift-rounding
// at 31 saturating lanes so the raw partial stays within the mesh's 20-bit
// `spatialArrayOutputType` envelope, plus a +127/-128 mix — see the case_neg128
// comment for why a full 32-lane -128 block is one LSB out of envelope); alternating
// signs with small scales (negative shift ⇒ nearest-even right-shift rounding);
// random valid exponents (mixed positive/negative shifts); K-tail (K=40, second
// block only 8 valid K lanes, the rest zero-padded — exercises tail masking); and a
// golden-only 0xff-reject check (loading 0xff into hardware would fire the
// MXScaleSRAM assert).
//
// Build prerequisite: compile against gemmini_params_mxint8_dim32.h (MX_ENABLED=1).
// Scope: single output tile (M = N = DIM); partial M/N tiles are a later case.

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

static uint32_t lcg = 0x13579bdfu;
static int rnd(int lo, int hi) { // uniform int in [lo, hi]
  lcg = lcg * 1664525u + 1013904223u;
  return lo + (int)((lcg >> 9) % (uint32_t)(hi - lo + 1));
}

static elem_t a_payload[M][K_MAX] row_align(1);
static elem_t b_payload[K_MAX][N] row_align(1);
static mx_scale_t a_scale[M][KB_MAX] __attribute__((aligned(64)));
static mx_scale_t b_scale[KB_MAX][N] __attribute__((aligned(64)));
static acc_t c_hw[M][N] row_align(1);
static acc_t gold[M][N];

static void mxint8_hw_matmul(size_t k, size_t kb) {
  const uint32_t acc_base = (1u << (ADDR_LEN - 1)) | (1u << (ADDR_LEN - 3));
  const uint32_t acc_accum = (1u << (ADDR_LEN - 2));
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

// Zero everything, then a case fills the valid region. Zeroing the K-tail
// padding lets the full-DIM block mvin carry exact zeros past the valid K.
static void clear_inputs(void) {
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

static void build_gold_uniform_raw(const int64_t raw[KB_MAX], size_t kb) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      int64_t acc = 0;
      for (size_t b = 0; b < kb; b++) {
        acc += mxint8_scale_raw_block(raw[b],
                                      mxint8_e8m0_decode(a_scale[i][b]),
                                      mxint8_e8m0_decode(b_scale[b][j]));
      }
      gold[i][j] = mxint8_saturate_acc(acc);
    }
  }
}

static void build_gold_parity_raw(int64_t raw_same_parity, int64_t raw_diff_parity) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      const int64_t raw = ((i + j) & 1) ? raw_diff_parity : raw_same_parity;
      gold[i][j] = mxint8_saturate_acc(mxint8_scale_raw_block(
          raw, mxint8_e8m0_decode(a_scale[i][0]), mxint8_e8m0_decode(b_scale[0][j])));
    }
  }
}

static int check(const char *label, size_t k, size_t kb) {
  printf("%-10s K=%d: hw begin\n", label, (int)k);
  mxint8_hw_matmul(k, kb);
  printf("%-10s K=%d: compare begin\n", label, (int)k);
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
    printf("%-10s K=%d: FAIL %d mism; first [%d][%d] hw=%d gold=%d\n",
           label, (int)k, mism, fi, fj, (int)c_hw[fi][fj], (int)gold[fi][fj]);
    return 1;
  }
  printf("%-10s K=%d: PASS (c[0][0]=%d c[1][2]=%d)\n",
         label, (int)k, (int)c_hw[0][0], (int)c_hw[1][2]);
  return 0;
}

static int case_zero(void) {
  clear_inputs();           // all payloads 0, scales valid ⇒ C must be all 0
  const int64_t raw[KB_MAX] = {0, 0};
  build_gold_uniform_raw(raw, 1);
  return check("zero", MX_BLOCK_SIZE, 1);
}

// Uniform payloads (A=`av`, B=`bv`), uniform scales eA=eB=ex ⇒ shift = 2*ex-12.
// raw = k*av*bv; with a large shift this saturates the int32 accumulator (sign of
// av*bv selects INT32_MAX vs INT32_MIN).
static int case_saturate(const char *label, elem_t av, elem_t bv, int ex) {
  clear_inputs();
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < MX_BLOCK_SIZE; kk++) { a_payload[i][kk] = av; }
    a_scale[i][0] = mxint8_e8m0_encode(ex);
  }
  for (size_t kk = 0; kk < MX_BLOCK_SIZE; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = bv; }
  }
  for (size_t j = 0; j < N; j++) { b_scale[0][j] = mxint8_e8m0_encode(ex); }
  const int64_t raw[KB_MAX] = {(int64_t)MX_BLOCK_SIZE * (int64_t)av * (int64_t)bv, 0};
  build_gold_uniform_raw(raw, 1);
  return check(label, MX_BLOCK_SIZE, 1);
}

static int case_altsign_round(void) {
  clear_inputs();
  // Alternating-sign payloads, small scales (eA=eB=2 ⇒ shift = -8: a nearest-even
  // right shift), so the per-block rounding path is exercised.
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < MX_BLOCK_SIZE; kk++) {
      a_payload[i][kk] = ((i + kk) & 1) ? -100 : 100;
    }
    a_scale[i][0] = mxint8_e8m0_encode(2);
  }
  for (size_t kk = 0; kk < MX_BLOCK_SIZE; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = ((kk + j) & 1) ? -123 : 123; }
  }
  for (size_t j = 0; j < N; j++) { b_scale[0][j] = mxint8_e8m0_encode(2); }
  const int64_t raw_same = (int64_t)MX_BLOCK_SIZE * 100 * 123;
  const int64_t raw_diff = -raw_same;
  build_gold_parity_raw(raw_same, raw_diff);
  return check("altsign", MX_BLOCK_SIZE, 1);
}

// `lanes` (<= MX_BLOCK_SIZE-1) lanes of -128 in both A and B, the rest zero, under
// uniform scale ex. raw = lanes * (-128)^2 = lanes * 16384, kept within the mesh's
// 20-bit `spatialArrayOutputType` envelope [-2^19, 2^19-1] = [-524288, 524287]. A
// *full* 32-lane -128 block reaches 32*128^2 = 2^19 = 524288, one LSB past the +524287
// max, so it exceeds the RES-OPT-narrowed raw width and is outside the supported
// envelope; the conformant packer never emits -128 (OCP 5.3.4 "may"; OCP_CONFORMANCE.md
// row 7). 31 lanes give 507904, a near-max representable -128 block.
static int case_neg128(const char *label, size_t lanes, int ex) {
  clear_inputs();
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < lanes; kk++) { a_payload[i][kk] = -128; }
    a_scale[i][0] = mxint8_e8m0_encode(ex);
  }
  for (size_t kk = 0; kk < lanes; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = -128; }
  }
  for (size_t j = 0; j < N; j++) { b_scale[0][j] = mxint8_e8m0_encode(ex); }
  const int64_t raw[KB_MAX] = {(int64_t)lanes * 128 * 128, 0};
  build_gold_uniform_raw(raw, 1);
  return check(label, MX_BLOCK_SIZE, 1);
}

static int case_neg128_mixed(void) {
  clear_inputs();
  // -128 interleaved with +127 under random valid exponents: the raw partial mixes
  // the spec's unused-by-encoders -2.0 encoding with max normal across both sign
  // regimes. The structured alternation keeps raw in [-520192, 520208], inside the
  // 20-bit mesh envelope.
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < MX_BLOCK_SIZE; kk++) {
      a_payload[i][kk] = ((i + kk) & 1) ? -128 : 127;
    }
    a_scale[i][0] = mxint8_e8m0_encode(rnd(-10, 10));
  }
  for (size_t kk = 0; kk < MX_BLOCK_SIZE; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = ((kk + j) & 1) ? -128 : 127; }
  }
  for (size_t j = 0; j < N; j++) { b_scale[0][j] = mxint8_e8m0_encode(rnd(-10, 10)); }
  const int64_t raw_same = 16 * ((int64_t)127 * 127) + 16 * ((int64_t)128 * 128);
  const int64_t raw_diff = -32 * ((int64_t)127 * 128);
  build_gold_parity_raw(raw_same, raw_diff);
  return check("neg128mix", MX_BLOCK_SIZE, 1);
}

static int case_random_exp(void) {
  clear_inputs();
  // Deterministic payloads + random valid E8M0 exponents in [-18,18] ⇒ a spread
  // of positive and negative shifts without spending target cycles in a full
  // matrix golden loop.
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < MX_BLOCK_SIZE; kk++) { a_payload[i][kk] = 1; }
    a_scale[i][0] = mxint8_e8m0_encode(rnd(-18, 18));
  }
  for (size_t kk = 0; kk < MX_BLOCK_SIZE; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = 1; }
  }
  for (size_t j = 0; j < N; j++) { b_scale[0][j] = mxint8_e8m0_encode(rnd(-18, 18)); }
  const int64_t raw[KB_MAX] = {MX_BLOCK_SIZE, 0};
  build_gold_uniform_raw(raw, 1);
  return check("randexp", MX_BLOCK_SIZE, 1);
}

static int case_ktail(void) {
  clear_inputs();
  // K = 40: block 0 full (32 lanes), block 1 has 8 valid K lanes [32,40) and the
  // rest zero-padded (clear_inputs already zeroed [40,64)). Per-block scales differ.
  const size_t k = 40, kb = 2;
  for (size_t i = 0; i < M; i++) {
    for (size_t kk = 0; kk < MX_BLOCK_SIZE; kk++) { a_payload[i][kk] = 1; }
    for (size_t kk = MX_BLOCK_SIZE; kk < k; kk++) { a_payload[i][kk] = 2; }
    a_scale[i][0] = mxint8_e8m0_encode(6 + (int)(i % 4));
    a_scale[i][1] = mxint8_e8m0_encode(5 + (int)(i % 3));
  }
  for (size_t kk = 0; kk < MX_BLOCK_SIZE; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = 1; }
  }
  for (size_t kk = MX_BLOCK_SIZE; kk < k; kk++) {
    for (size_t j = 0; j < N; j++) { b_payload[kk][j] = 3; }
  }
  for (size_t j = 0; j < N; j++) {
    b_scale[0][j] = mxint8_e8m0_encode(6 + (int)(j % 3));
    b_scale[1][j] = mxint8_e8m0_encode(4 + (int)(j % 2));
  }
  const int64_t raw[KB_MAX] = {MX_BLOCK_SIZE, 8 * 2 * 3};
  build_gold_uniform_raw(raw, kb);
  return check("ktail", k, kb);
}

// 0xff (E8M0 NaN) is rejected: the golden returns -1, and loading it into the
// MXScaleSRAM would fire a hardware assert, so this is checked software-only.
static int case_nan_reject(void) {
  clear_inputs();
  for (size_t i = 0; i < M; i++) { a_payload[i][0] = 1; }
  for (size_t j = 0; j < N; j++) { b_payload[0][j] = 1; }
  a_scale[3][0] = (mx_scale_t)0xff;
  const int rc = mxint8_ref_gemm_acc(&a_payload[0][0], &b_payload[0][0],
                                     &a_scale[0][0], &b_scale[0][0], &gold[0][0],
                                     4, 1, 1, K_MAX, N, N, KB_MAX, N);
  const int ok = (rc == -1);
  printf("nan_reject K=32: %s (golden rc=%d, expected -1)\n", ok ? "PASS" : "FAIL", rc);
  return ok ? 0 : 1;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int bad = 0;
  bad |= case_zero();
  bad |= case_saturate("max_pos32", 127, 127, 13);  // shift 14, +raw ⇒ INT32_MAX
  bad |= case_saturate("max_neg32", 127, -127, 13); // shift 14, -raw ⇒ INT32_MIN
  bad |= case_saturate("max_pos64", 127, 127, 31);  // shift 50: 64-bit-overflow path ⇒ INT32_MAX
  // -128 payloads (consumer side of the spec-5.3.4 "may leave -2 unused" rule), kept
  // within the 20-bit mesh raw envelope via 31 saturating lanes (raw = 31*16384 =
  // 507904). The conformant packer never emits -128.
  bad |= case_neg128("neg128_exact", 31, 6);   // shift 0:  C = 507904 exactly
  bad |= case_neg128("neg128_sat",   31, 13);  // shift 14: 507904<<14 > INT32_MAX ⇒ INT32_MAX
  bad |= case_neg128("neg128_round", 31, -2);  // shift -16: 507904>>16 RNE = 8
  bad |= case_neg128_mixed();
  bad |= case_altsign_round();
  bad |= case_random_exp();
  bad |= case_ktail();
  bad |= case_nan_reject();

  if (bad) {
    printf("mxint8_corner: FAIL\n");
    exit(1);
  }
  printf("mxint8_corner: PASS\n");
  exit(0);
}

#endif // MX_ENABLED
