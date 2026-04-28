// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

#include "remote.h"
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>  

#include "sdkl.h"

/*!
 @brief to get SDKL version string from  sdkl_npu_get_version()
*/
char version[SDKL_VERSION_STR_LEN];

#define MAX_VAL 5 // Random values are in [0, MAX_VAL]
#define N_ROW   1024
#define N_COL   3072
#define N_INNER 8192

/// @brief Utility macro to check SDKL returns 0 and, if an error occured,
///        pretty-print the \ref error and exit on EXIT_FAILURE
#define SDKL_CHECK(x) \
  do { \
    if ((x) != 0) { \
      printf("Line = %d, nErr = %d\n", __LINE__, x); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

// ----------------------------------------------------------------------------
// Basic loop version
// ----------------------------------------------------------------------------

// Matrix multiplication A = X * W^T

__attribute__((noinline)) void matmul(
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  _Float16* A,       // A[n_row][n_col]
  const _Float16* X, // X[n_row][n_inner]
  const _Float16* W  // W[n_col][n_inner]
) {
  float dot = 0.0f;
  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col; j++) {
      dot = 0.0f;
      for (size_t k = 0; k < n_inner; k++) {
        dot += (float)X[i * n_inner + k] * (float)W[j * n_inner + k];
      }
      A[i * n_col + j] = (_Float16)dot;
    }
  }
}

/*!
  @brief
  Compares SDKL API result vs Standard C reference. Tolerates 0.1% error
*/
bool sdkl_vector_check_f16(size_t size, _Float16* ref, _Float16* vec) {
  bool res = true;
  for (int32_t i = 0; i < size; i++) {
    _Float16 diff;
    _Float16 diff_0dot001percent = fabsf(ref[i] / (_Float16)1000.0f);

    if (isnan((float)ref[i])) {
      res = false;
      printf("ERROR ref[%ld] = %f vec[%ld] = %f\n", (long)i, (float)ref[i], (long)i, (float)vec[i]);
      break;
    }

    if (isinf((float)vec[i])) {
      res = false;
      printf("ERROR ref[%ld] = %f vec[%ld] = %f\n", (long)i, (float)ref[i], (long)i, (float)vec[i]);
      break;
    }
    diff = fabsf(ref[i] - vec[i]);
    if (diff > diff_0dot001percent) {
      res = false;
      printf("ERROR ref[%ld] = %f vec[%ld] = %f\n", (long)i, (float)ref[i], (long)i, (float)vec[i]);
      break;
    }
  }
  return res;
}

static double elapsed(struct timeval start, struct timeval end) {
  long seconds, useconds;
  seconds  = end.tv_sec - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;
  return (seconds) + useconds / 1000000.;
}

int main() {
  struct timeval start, end;
  double time_reference = 0;
  int res               = true;
  int domain            = CDSP_DOMAIN_ID;

  _Float16* A_f16_cpu_reference = NULL;
  _Float16* A_f16_npu           = NULL;

  _Float16* X_f16_cpu = NULL;
  _Float16* X_f16_npu = NULL;

  _Float16* W_f16_cpu = NULL;
  _Float16* W_f16_npu = NULL;

  size_t A_f16_cpu_size = N_ROW * N_COL * sizeof(*A_f16_cpu_reference);
  size_t X_f16_cpu_size = N_ROW * N_INNER * sizeof(*X_f16_cpu);
  size_t W_f16_cpu_size = N_COL * N_INNER * sizeof(*W_f16_cpu);

  // Initialize SDKL
  SDKL_CHECK(sdkl_npu_initialize(domain, NULL, NULL));

  SDKL_CHECK(sdkl_npu_get_version(domain, version));

  printf("SDKL Version: %s\n", version);

  A_f16_cpu_reference = malloc(A_f16_cpu_size);
  X_f16_cpu           = malloc(X_f16_cpu_size);
  W_f16_cpu           = malloc(W_f16_cpu_size);

  SDKL_CHECK(sdkl_npu_alloc(W_f16_cpu_size, (void**)&W_f16_npu));
  SDKL_CHECK(sdkl_npu_alloc(X_f16_cpu_size, (void**)&X_f16_npu));
  SDKL_CHECK(sdkl_npu_alloc(A_f16_cpu_size, (void**)&A_f16_npu));

  // Initialization by random values
  srand(42); //(unsigned int)time(NULL));
  printf("SDKL Test Start:\n");

  for (size_t i = 0; i < N_ROW; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      X_f16_cpu[i * N_INNER + j] = ((float)1.0f) * ((float)rand() / (float)RAND_MAX);
      X_f16_npu[i * N_INNER + j] = X_f16_cpu[i * N_INNER + j];
    }
  }
  for (size_t i = 0; i < N_COL; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      W_f16_cpu[i * N_INNER + j] = ((float)1.0f) * ((float)rand() / (float)RAND_MAX);
      W_f16_npu[i * N_INNER + j] = W_f16_cpu[i * N_INNER + j];
    }
  }

  for (size_t i = 0; i < N_ROW * N_COL; i++) {
    A_f16_cpu_reference[i] = 0.0f;
    A_f16_npu[i]           = 0.0f;
  }

  // Run and profile reference C code
  gettimeofday(&start, NULL);
  matmul(N_ROW, N_COL, N_INNER, A_f16_cpu_reference, X_f16_cpu, W_f16_cpu);
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("CPU single thread, runs %-.5lf s\n", time_reference);

  // Perform Weights layout
  gettimeofday(&start, NULL);
  SDKL_CHECK(sdkl_cpu_rm_to_wh_f16_inplace(N_COL, N_INNER, W_f16_npu));
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Weights data-layout runs %-.5lf s\n", time_reference);

  gettimeofday(&start, NULL);

  SDKL_CHECK(sdkl_npu_mm_f16f16_f16(domain, N_ROW, N_COL, N_INNER, A_f16_npu, X_f16_npu, W_f16_npu));

  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("NPU runs %-.5lf s\n", time_reference);

  // --------------------------------------------------------------------------

  // Check if the result differs from the reference
  res = sdkl_vector_check_f16(N_ROW * N_COL, A_f16_cpu_reference, A_f16_npu);

  if (res) {
    printf("Test Passed\n");
  } else {
    printf("Test Failed\n");
  }

  free(A_f16_cpu_reference);
  free(X_f16_cpu);
  free(W_f16_cpu);

  SDKL_CHECK(sdkl_npu_free(W_f16_npu));
  SDKL_CHECK(sdkl_npu_free(A_f16_npu));
  SDKL_CHECK(sdkl_npu_free(X_f16_npu));

  // Finalize & cleanup SDKL
  SDKL_CHECK(sdkl_npu_finalize(domain));

  return 0;
}
