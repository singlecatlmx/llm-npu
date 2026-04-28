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

/// @brief Utility macro to check SDKL returns 0 and, if an error occurred,
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

__attribute__((noinline)) void matmul_ui8i8_i32_rm(
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  int32_t* A, // A[n_row][n_col]
  uint8_t* X, // X[n_row][n_inner]
  int8_t* W   // W[n_col][n_inner]
) {
  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col; j++) {
      A[i * n_col + j] = 0;
      for (size_t k = 0; k < n_inner; k++) {
        A[i * n_col + j] += (int32_t)X[i * n_inner + k] * (int32_t)W[j * n_inner + k];
      }
    }
  }
}

bool vector_validation(size_t n_row, size_t n_col, int32_t* A_i32_cpu_a8w8, int32_t* A_i32_npu_a8w8) {
  bool res = true;
  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col; j++) {
      if (A_i32_cpu_a8w8[j + n_col * i] != A_i32_npu_a8w8[j + n_col * i]) {
        res = false;
        return res;
      }
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
  bool res              = true;
  int domain            = CDSP_DOMAIN_ID;

  int32_t* A_i32_cpu_reference = NULL;
  int32_t* A_i32_npu           = NULL;

  uint8_t* X_u8_cpu = NULL;
  uint8_t* X_u8_npu = NULL;

  int8_t* W_i8_cpu = NULL;
  int8_t* W_i8_npu = NULL;

  size_t A_i32_cpu_size = N_ROW * N_COL * sizeof(*A_i32_cpu_reference);
  size_t X_u8_cpu_size  = N_ROW * N_INNER * sizeof(*X_u8_cpu);
  size_t W_i8_cpu_size  = N_COL * N_INNER * sizeof(*W_i8_cpu);

  // Initialize SDKL
  SDKL_CHECK(sdkl_npu_initialize(domain, NULL, NULL));

  SDKL_CHECK(sdkl_npu_get_version(domain, version));

  printf("SDKL Version: %s\n", version);

  A_i32_cpu_reference = malloc(A_i32_cpu_size);
  X_u8_cpu            = malloc(X_u8_cpu_size);
  W_i8_cpu            = malloc(W_i8_cpu_size);

  SDKL_CHECK(sdkl_npu_alloc(W_i8_cpu_size, (void**)&W_i8_npu));
  SDKL_CHECK(sdkl_npu_alloc(X_u8_cpu_size, (void**)&X_u8_npu));
  SDKL_CHECK(sdkl_npu_alloc(A_i32_cpu_size, (void**)&A_i32_npu));

  // Initialization by random values
  srand(42); //(unsigned int)time(NULL));
  printf("SDKL Test Start:\n");

  for (size_t i = 0; i < N_ROW; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      X_u8_cpu[i * N_INNER + j] = (uint8_t)rand() % 23;
      X_u8_npu[i * N_INNER + j] = X_u8_cpu[i * N_INNER + j];
    }
  }
  for (size_t i = 0; i < N_COL; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      W_i8_cpu[i * N_INNER + j] = (int8_t)(rand() % 38) - 32;
      W_i8_npu[i * N_INNER + j] = W_i8_cpu[i * N_INNER + j];
    }
  }

  for (size_t i = 0; i < N_ROW * N_COL; i++) {
    A_i32_cpu_reference[i] = 0;
    A_i32_npu[i]           = 0;
  }

  // Run and profile reference C code
  gettimeofday(&start, NULL);
  matmul_ui8i8_i32_rm(N_ROW, N_COL, N_INNER, A_i32_cpu_reference, X_u8_cpu, W_i8_cpu);
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("CPU single thread, runs %-.5lf s\n", time_reference);

  // Perform Weights layout
  gettimeofday(&start, NULL);
  SDKL_CHECK(sdkl_cpu_rm_to_wh_i8_inplace(N_COL, N_INNER, W_i8_npu));
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Weights data-layout runs %-.5lf s\n", time_reference);

  gettimeofday(&start, NULL);
  SDKL_CHECK(sdkl_npu_mm_u8i8_i32(domain, N_ROW, N_COL, N_INNER, A_i32_npu, X_u8_npu, W_i8_npu));
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("NPU runs %-.5lf s\n", time_reference);

  // --------------------------------------------------------------------------

  // Check if the result differs from the reference
  res = vector_validation(N_ROW, N_COL, A_i32_cpu_reference, A_i32_npu);

  if (res) {
    printf("Test Passed\n");
  } else {
    printf("Test Failed\n");
  }

  free(A_i32_cpu_reference);
  free(X_u8_cpu);
  free(W_i8_cpu);

  SDKL_CHECK(sdkl_npu_free(W_i8_npu));
  SDKL_CHECK(sdkl_npu_free(A_i32_npu));
  SDKL_CHECK(sdkl_npu_free(X_u8_npu));

  // Finalize & cleanup SDKL
  SDKL_CHECK(sdkl_npu_finalize(domain));

  return 0;
}
