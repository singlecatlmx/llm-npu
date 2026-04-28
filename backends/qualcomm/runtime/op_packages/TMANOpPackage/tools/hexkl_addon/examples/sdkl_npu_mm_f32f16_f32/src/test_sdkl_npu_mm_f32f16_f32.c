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
  float* A,   // A[n_row][n_col]
  float* X,   // X[n_row][n_inner]
  _Float16* W // W[n_col][n_inner]
) {
  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col; j++) {
      A[i * n_col + j] = 0.;
      for (size_t k = 0; k < n_inner; k++) {
        A[i * n_col + j] += X[i * n_inner + k] * W[j * n_inner + k];
      }
    }
  }
}

/*!
  @brief
  Compares SDKL API result vs Standard C reference. Tolerates 0.1% error
*/
bool sdkl_vector_check_f32(size_t size, float* ref, float* vec) {
  bool res = true;
  for (int32_t i = 0; i < size; i++) {
    float diff;
    float diff_0dot001percent = fabsf(ref[i] / (float)1000.0f);

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

  float* A_f32_cpu_reference = NULL;
  float* A_f32_cpu_hexkl_npu = NULL;

  float* X_f32_cpu = NULL; // Input for CPU and NPU in this example

  _Float16* W_f16_cpu = NULL;
  _Float16* W_f16_npu = NULL;

  size_t A_f32_cpu_size = N_ROW * N_COL * sizeof(*A_f32_cpu_reference);
  size_t X_f32_cpu_size = N_ROW * N_INNER * sizeof(*X_f32_cpu);
  size_t W_f16_cpu_size = N_COL * N_INNER * sizeof(*W_f16_cpu);
  size_t W_f16_npu_size = N_COL * N_INNER * sizeof(*W_f16_npu);

  // Initialize SDKL
  SDKL_CHECK(sdkl_npu_initialize(domain, NULL, NULL));

  SDKL_CHECK(sdkl_npu_get_version(domain, version));

  printf("SDKL Version: %s\n", version);

  A_f32_cpu_reference = malloc(A_f32_cpu_size);
  X_f32_cpu           = malloc(X_f32_cpu_size);
  W_f16_cpu           = malloc(W_f16_cpu_size);
  A_f32_cpu_hexkl_npu = malloc(A_f32_cpu_size);

  SDKL_CHECK(sdkl_npu_alloc(W_f16_npu_size, (void**)&W_f16_npu));

  // Initialization by random values
  srand(42); //(unsigned int)time(NULL));
  printf("SDKL Test Start:\n");

  for (size_t i = 0; i < N_ROW; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      X_f32_cpu[i * N_INNER + j] = ((float)1.0f) * ((float)rand() / (float)RAND_MAX);
    }
  }
  for (size_t i = 0; i < N_COL; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      W_f16_cpu[i * N_INNER + j] = ((float)1.0f) * ((float)rand() / (float)RAND_MAX);
      W_f16_npu[i * N_INNER + j] = W_f16_cpu[i * N_INNER + j];
    }
  }

  for (size_t i = 0; i < N_ROW * N_COL; i++) {
    A_f32_cpu_reference[i] = 0.f;
    A_f32_cpu_hexkl_npu[i] = 0.f;
  }

  // Run and profile reference C code
  gettimeofday(&start, NULL);
  matmul(N_ROW, N_COL, N_INNER, A_f32_cpu_reference, X_f32_cpu, W_f16_cpu);
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("CPU single thread runs %-.5lf s\n", time_reference);

  // Perform Weights layout
  gettimeofday(&start, NULL);
  SDKL_CHECK(sdkl_cpu_rm_to_wh_f16_inplace(N_COL, N_INNER, W_f16_npu));
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Weights data-layout runs %-.5lf s\n", time_reference);

  gettimeofday(&start, NULL);

  SDKL_CHECK(sdkl_npu_mm_f32f16_f32(domain, N_ROW, N_COL, N_INNER, A_f32_cpu_hexkl_npu, X_f32_cpu, W_f16_npu));

  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("NPU runs %-.5lf s\n", time_reference);

  // --------------------------------------------------------------------------

  // Check if the result differs from the reference
  res = sdkl_vector_check_f32(N_ROW * N_COL, A_f32_cpu_reference, A_f32_cpu_hexkl_npu);

  if (res) {
    printf("Test Passed\n");
  } else {
    printf("Test Failed\n");
  }

  // Cleanup
  free(A_f32_cpu_reference);
  free(A_f32_cpu_hexkl_npu);
  free(X_f32_cpu);
  free(W_f16_cpu);

  SDKL_CHECK(sdkl_npu_free(W_f16_npu));

  // Finalize & cleanup SDKL
  SDKL_CHECK(sdkl_npu_finalize(domain));

  return 0;
}
