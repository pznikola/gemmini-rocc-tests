// See LICENSE for license details.
//
// MXINT8 host reference harness: runs the *in-repo* packer (`mxint8_pack.h`)
// and golden GEMM (`mxint8_golden.h`) on the host, so external cross-checks
// (`mxint8_external_diff.py`) diff the actual project code rather than a
// reimplementation.
//
// Usage:
//   mxint8_host_ref <m> <n> <k> <a_f32.bin> <b_f32.bin> <out_prefix>
//
// Inputs are dense row-major little-endian fp32: A[m][k], B[k][n]. Outputs
// (raw little-endian binaries, kb = ceil(k/32)):
//   <out_prefix>.a_payload.bin  int8   m*k
//   <out_prefix>.b_payload.bin  int8   k*n
//   <out_prefix>.a_scale.bin    uint8  m*kb
//   <out_prefix>.b_scale.bin    uint8  kb*n
//   <out_prefix>.c.bin          int32  m*n   (golden `mxint8_ref_gemm_acc`)
//
// Build (gemmini_params.h must be the staged MX DIM=32 header, exactly as the
// baremetal test build stages it; the Python driver does this automatically):
//   gcc -O2 -I <staging_dir> tools/mxint8_host_ref.c -o mxint8_host_ref

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/gemmini_params.h"
#include "include/mxint8_golden.h"
#include "include/mxint8_pack.h"

/**
 * Read exactly `count` items of `size` bytes from a binary file.
 *
 * @return 0 on success, nonzero (with a message on stderr) otherwise.
 */
static int read_bin(const char *path, void *buf, size_t size, size_t count) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    fprintf(stderr, "mxint8_host_ref: cannot open %s\n", path);
    return 1;
  }
  const size_t got = fread(buf, size, count, f);
  fclose(f);
  if (got != count) {
    fprintf(stderr, "mxint8_host_ref: %s: expected %zu items, got %zu\n", path, count, got);
    return 1;
  }
  return 0;
}

/**
 * Write `count` items of `size` bytes to `<prefix><suffix>`.
 *
 * @return 0 on success, nonzero (with a message on stderr) otherwise.
 */
static int write_bin(const char *prefix, const char *suffix, const void *buf,
                     size_t size, size_t count) {
  char path[4096];
  snprintf(path, sizeof(path), "%s%s", prefix, suffix);
  FILE *f = fopen(path, "wb");
  if (f == NULL) {
    fprintf(stderr, "mxint8_host_ref: cannot create %s\n", path);
    return 1;
  }
  const size_t put = fwrite(buf, size, count, f);
  fclose(f);
  if (put != count) {
    fprintf(stderr, "mxint8_host_ref: short write to %s\n", path);
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: %s <m> <n> <k> <a_f32.bin> <b_f32.bin> <out_prefix>\n", argv[0]);
    return 2;
  }

  const size_t m = (size_t)strtoul(argv[1], NULL, 0);
  const size_t n = (size_t)strtoul(argv[2], NULL, 0);
  const size_t k = (size_t)strtoul(argv[3], NULL, 0);
  if (m == 0 || n == 0 || k == 0) {
    fprintf(stderr, "mxint8_host_ref: m, n, k must be positive\n");
    return 2;
  }
  const size_t kb = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;

  float *a = malloc(m * k * sizeof(float));
  float *b = malloc(k * n * sizeof(float));
  elem_t *a_payload = malloc(m * k * sizeof(elem_t));
  elem_t *b_payload = malloc(k * n * sizeof(elem_t));
  mx_scale_t *a_scale = malloc(m * kb * sizeof(mx_scale_t));
  mx_scale_t *b_scale = malloc(kb * n * sizeof(mx_scale_t));
  acc_t *c = malloc(m * n * sizeof(acc_t));
  if (!a || !b || !a_payload || !b_payload || !a_scale || !b_scale || !c) {
    fprintf(stderr, "mxint8_host_ref: out of memory\n");
    return 1;
  }

  if (read_bin(argv[4], a, sizeof(float), m * k) != 0 ||
      read_bin(argv[5], b, sizeof(float), k * n) != 0) {
    return 1;
  }

  mxint8_pack_a(a, m, k, k, kb, a_payload, a_scale);
  mxint8_pack_b(b, k, n, n, n, b_payload, b_scale);

  const int rc = mxint8_ref_gemm_acc(a_payload, b_payload, a_scale, b_scale, c,
                                     m, n, k, k, n, n, kb, n);
  if (rc != 0) {
    fprintf(stderr, "mxint8_host_ref: golden rejected the inputs (rc=%d)\n", rc);
    return 1;
  }

  if (write_bin(argv[6], ".a_payload.bin", a_payload, sizeof(elem_t), m * k) != 0 ||
      write_bin(argv[6], ".b_payload.bin", b_payload, sizeof(elem_t), k * n) != 0 ||
      write_bin(argv[6], ".a_scale.bin", a_scale, sizeof(mx_scale_t), m * kb) != 0 ||
      write_bin(argv[6], ".b_scale.bin", b_scale, sizeof(mx_scale_t), kb * n) != 0 ||
      write_bin(argv[6], ".c.bin", c, sizeof(acc_t), m * n) != 0) {
    return 1;
  }

  return 0;
}
