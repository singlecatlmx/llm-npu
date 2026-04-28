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
/*!
 @brief Utility macro to check SDKL returns 0 and, if an error occurred,
        pretty-print the \ref error and exit on EXIT_FAILURE
*/
#define SDKL_CHECK(x) \
  do { \
    if ((x) != 0) { \
      printf("Line = %d, nErr = %d\n", __LINE__, x); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

/*!
  @brief
  Compares SDKL API FP32 result vs Standard C reference. Tolerates 0.1% error
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

/*!
  @brief
  Compares SDKL API FP16 result vs Standard C reference. Tolerates 0.1% error
*/
bool sdkl_vector_check_f16(size_t size, _Float16* ref, _Float16* vec) {
  bool res = true;
  for (int32_t i = 0; i < size; i++) {
    float diff;
    float diff_0dot001percent = fabsf((float)ref[i] / (float)1000.0f);

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
    diff = fabsf((float)ref[i] - (float)vec[i]);
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

sdkl_tensor_t right_mat; /*used for weights*/
sdkl_tensor_t left_mat;  /*used for activation*/
sdkl_tensor_t res_mat;
sdkl_tensor_t res_cpu_mat;
int platform_npu              = SDKL_PLATFORM_NPU0;
int platform_cpu              = SDKL_PLATFORM_CPU;
float* X_f32                  = NULL; // Input for CPU and NPU in this example
_Float16* X_f16               = NULL;
_Float16* X_f16_npu           = NULL;
_Float16* W_f16               = NULL;
_Float16* W_f16_npu           = NULL;
uint8_t* X_ui8                = NULL; // For scenario 6 and 7
int8_t* W_i8                  = NULL; // For scenario 6 and 7
uint8_t* W_ui8_npu_scenario_6 = NULL; // For scenario 6 NPU weights
int8_t* W_i8_npu_scenario_7   = NULL; // For scenario 7 NPU weights
double time_reference         = 0;
uint64_t weight_offset        = 15;
uint64_t res_offset           = 66;
uint64_t x_offset             = 114;
uint64_t res_stride           = N_COL * 2;
struct timeval start, end;

/*!
  @brief
  Scenario 1: Matrix multiplication using SDKL NPU with mixed precision and pre-transposed HMX-arranged weights.

  This scenario performs matrix multiplication using the SDKL NPU with:
  - FP16 weights in pre-transposed HMX layout (`SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX`)
  - FP32 activations in standard row-major layout (`SDKL_LAYOUT_2D_ROW_MAJOR`)
  - FP32 output in standard row-major layout

  This function sets up three tensors for NPU-based matrix multiplication:
  - `right_mat`: FP16 weights, pre-transposed and arranged for HMX layout.
  - `left_mat`: FP32 input matrix, unquantized and in standard row-major layout.
  - `res_mat`: FP32 output matrix in continuous row-major layout.

  The right matrix uses `SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX` layout, indicating that the weights are already
  laid out in a transposed format optimized for HMX hardware. No additional transposition is performed.

  The function configures tensor metadata including dimensions, strides, quantization type,
  and data layout, then invokes `sdkl_mm_tensor()` to perform the multiplication.

  Execution time is measured using `gettimeofday()` and printed to stdout.

  @see sdkl_mm_tensor
*/

float* A_f32_ref_scenario_1  = NULL; /* Reference output */
float* A_f32_sdkl_scenario_1 = NULL; /* SDKL output */

/*!
  @brief
  Reference code for matrix multiplication A = X * W^T
 */
__attribute__((noinline)) void matmul_ref_scenario_1(
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  float* A,         // A[n_row][n_col]
  const float* X,   // X[n_row][n_inner]
  const _Float16* W // W[n_col][n_inner]
) {
  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col; j++) {
      float acc = 0.0f;
      for (size_t k = 0; k < n_inner; k++) {
        acc += X[i * n_inner + k] * W[j * n_inner + k];
      }
      A[i * n_col + j] = acc;
    }
  }
}

/*!
  @brief
  NPU-offload, tensor-based code for matrix multiplication A = X * W^T
*/
void matmul_sdkl_scenario_1() {
  right_mat.data          = (void*)W_f16_npu;
  right_mat.ndims         = 2;
  right_mat.dims[0]       = N_INNER;
  right_mat.dims[1]       = N_COL;
  right_mat.num_elements  = N_INNER * N_COL + weight_offset;
  right_mat.is_continuous = 1;
  right_mat.quantization  = SDKL_QUANT_NONE;
  right_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX;
  right_mat.data_dtype    = SDKL_DTYPE_FP16;
  right_mat.data_offset   = 0;
  right_mat.strides[0]    = N_COL;
  right_mat.strides[1]    = 1;
  right_mat.data_offset   = weight_offset;

  left_mat.data          = (void*)X_f32;
  left_mat.ndims         = 2;
  left_mat.dims[0]       = N_ROW;
  left_mat.dims[1]       = N_INNER;
  left_mat.num_elements  = N_ROW * N_INNER + x_offset;
  left_mat.is_continuous = 1;
  left_mat.quantization  = SDKL_QUANT_NONE;
  left_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  left_mat.data_dtype    = SDKL_DTYPE_FP32;
  left_mat.data_offset   = x_offset;
  left_mat.strides[0]    = N_INNER;
  left_mat.strides[1]    = 1;

  res_mat.data          = A_f32_sdkl_scenario_1;
  res_mat.ndims         = 2;
  res_mat.dims[0]       = N_ROW;
  res_mat.dims[1]       = N_COL;
  res_mat.num_elements  = N_ROW * N_COL;
  res_mat.is_continuous = 1;
  res_mat.quantization  = SDKL_QUANT_NONE;
  res_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  res_mat.data_dtype    = SDKL_DTYPE_FP32;
  res_mat.data_offset   = 0;
  res_mat.strides[0]    = N_COL;
  res_mat.strides[1]    = 1;

  gettimeofday(&start, NULL);

  SDKL_CHECK(sdkl_mm_tensor(platform_npu, &res_mat, &left_mat, &right_mat));

  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 1, NPU runs %-.5lf s\n", time_reference);
}

/*!
  @brief
  Scenario 2: Matrix multiplication using SDKL NPU with mixed precision and strided output layout.

  This scenario performs matrix multiplication using the SDKL NPU with:
  - FP16 weights in pre-transposed HMX layout (`SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX`)
  - FP32 activations in standard row-major layout (`SDKL_LAYOUT_2D_ROW_MAJOR`)
  - FP32 output in strided row-major layout (non-continuous)

  This function sets up three tensors for NPU-based matrix multiplication:
  - `right_mat`: FP16 weights, pre-transposed and arranged for HMX layout.
  - `left_mat`: FP32 input matrix, unquantized and in standard row-major layout.
  - `res_mat`: FP32 output matrix with strided layout (`res_stride` between rows).

  Unlike Scenario 1, the result tensor is not continuous in memory. Instead, it uses a strided layout
  to accommodate alignment or padding requirements. The number of elements in the result buffer is
  computed precisely based on the maximum index accessed during matrix multiplication: `(N_ROW - 1) * res_stride +
  N_COL`.

  The function configures tensor metadata including dimensions, strides, quantization type,
  and data layout, then invokes `sdkl_mm_tensor()` to perform the multiplication.

  Execution time is measured using `gettimeofday()` and printed to stdout.

  @see sdkl_mm_tensor
*/

float* A_f32_ref_scenario_2  = NULL; /* Reference output */
float* A_f32_sdkl_scenario_2 = NULL; /* SDKL output */

/*!
  @brief
  Reference code for matrix multiplication A = X * W^T
*/
__attribute__((noinline)) void matmul_ref_scenario_2(
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  size_t result_stride,
  float* A,         // A[n_row][n_col]
  const float* X,   // X[n_row][n_inner]
  const _Float16* W // W[n_col][n_inner]
) {
  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col; j++) {
      float acc = 0.0f;
      for (size_t k = 0; k < n_inner; k++) {
        acc += X[i * n_inner + k] * W[j * n_inner + k];
      }
      A[i * result_stride + j] = acc;
    }
  }
}

/*!
  @brief
  NPU-offload, tensor-based code for matrix multiplication A = X * W^T
*/
void matmul_sdkl_scenario_2() {
  right_mat.data          = (void*)W_f16_npu;
  right_mat.ndims         = 2;
  right_mat.dims[0]       = N_INNER;
  right_mat.dims[1]       = N_COL;
  right_mat.num_elements  = N_INNER * N_COL + weight_offset;
  right_mat.is_continuous = 1;
  right_mat.quantization  = SDKL_QUANT_NONE;
  right_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX;
  right_mat.data_dtype    = SDKL_DTYPE_FP16;
  right_mat.data_offset   = 0;
  right_mat.strides[0]    = N_COL;
  right_mat.strides[1]    = 1;
  right_mat.data_offset   = weight_offset;

  left_mat.data          = (void*)X_f32;
  left_mat.ndims         = 2;
  left_mat.dims[0]       = N_ROW;
  left_mat.dims[1]       = N_INNER;
  left_mat.num_elements  = N_ROW * N_INNER + x_offset;
  left_mat.is_continuous = 1;
  left_mat.quantization  = SDKL_QUANT_NONE;
  left_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  left_mat.data_dtype    = SDKL_DTYPE_FP32;
  left_mat.data_offset   = x_offset;
  left_mat.strides[0]    = N_INNER;
  left_mat.strides[1]    = 1;

  res_mat.data          = A_f32_sdkl_scenario_2;
  res_mat.ndims         = 2;
  res_mat.dims[0]       = N_ROW;
  res_mat.dims[1]       = N_COL;
  res_mat.num_elements  = (N_ROW - 1) * res_stride + N_COL;
  res_mat.is_continuous = 0;
  res_mat.quantization  = SDKL_QUANT_NONE;
  res_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  res_mat.data_dtype    = SDKL_DTYPE_FP32;
  res_mat.data_offset   = 0;
  res_mat.strides[0]    = res_stride;
  res_mat.strides[1]    = 1;

  gettimeofday(&start, NULL);

  SDKL_CHECK(sdkl_mm_tensor(platform_npu, &res_mat, &left_mat, &right_mat));

  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 2, NPU runs %-.5lf s\n", time_reference);
}

/*!
  @brief
  Scenario 3: Matrix multiplication using SDKL NPU with FP16 inputs and strided FP16 output layout.

  This scenario performs matrix multiplication using the SDKL NPU with:
  - FP16 weights in pre-transposed HMX layout (`SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX`)
  - FP16 activations in standard row-major layout (`SDKL_LAYOUT_2D_ROW_MAJOR`)
  - FP16 output in strided row-major layout (non-continuous)

  This function sets up three tensors for NPU-based matrix multiplication:
  - `right_mat`: FP16 weights, pre-transposed and arranged in HMX-optimized layout for efficient NPU access.
  - `left_mat`: FP16 input matrix, unquantized and stored in standard row-major layout.
  - `res_mat`: FP16 output matrix with strided layout (`res_stride` between rows).

  Unlike Scenarios 1 and 2, this scenario uses FP16 precision for both inputs and outputs. The result tensor
  is not stored continuously in memory but uses a strided layout to accommodate alignment or padding requirements.
  The number of elements in the result buffer is computed precisely based on the maximum index accessed during
  matrix multiplication: `(N_ROW - 1) * res_stride + N_COL`.

  The function configures tensor metadata including dimensions, strides, quantization type,
  and data layout, then invokes `sdkl_mm_tensor()` to perform the multiplication.

  Execution time is measured using `gettimeofday()` and printed to stdout.

  @see sdkl_mm_tensor
*/

_Float16* A_f16_ref_scenario_3  = NULL; /* Reference output */
_Float16* A_f16_sdkl_scenario_3 = NULL; /* SDKL output */

/*!
  @brief
  Reference code for Matrix multiplication A = X * W^T
*/
__attribute__((noinline)) void matmul_ref_scenario_3(
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  size_t result_stride,
  _Float16* A,       // A[n_row][n_col]
  const _Float16* X, // X[n_row][n_inner]
  const _Float16* W  // W[n_col][n_inner]
) {
  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col; j++) {
      float acc = 0.0f;
      for (size_t k = 0; k < n_inner; k++) {
        acc += (float)X[i * n_inner + k] * (float)W[j * n_inner + k];
      }
      A[i * result_stride + j] = (_Float16)acc;
    }
  }
}

/*!
  @brief
  NPU-offload, tensor-based code for Matrix multiplication A = X * W^T
*/
void matmul_sdkl_scenario_3() {
  right_mat.data          = (void*)W_f16_npu;
  right_mat.ndims         = 2;
  right_mat.dims[0]       = N_INNER;
  right_mat.dims[1]       = N_COL;
  right_mat.num_elements  = N_INNER * N_COL + weight_offset;
  right_mat.is_continuous = 1;
  right_mat.quantization  = SDKL_QUANT_NONE;
  right_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX;
  right_mat.data_dtype    = SDKL_DTYPE_FP16;
  right_mat.data_offset   = 0;
  right_mat.strides[0]    = N_COL;
  right_mat.strides[1]    = 1;
  right_mat.data_offset   = weight_offset;

  left_mat.data          = (void*)X_f16;
  left_mat.ndims         = 2;
  left_mat.dims[0]       = N_ROW;
  left_mat.dims[1]       = N_INNER;
  left_mat.num_elements  = N_ROW * N_INNER + x_offset;
  left_mat.is_continuous = 1;
  left_mat.quantization  = SDKL_QUANT_NONE;
  left_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  left_mat.data_dtype    = SDKL_DTYPE_FP16;
  left_mat.data_offset   = x_offset;
  left_mat.strides[0]    = N_INNER;
  left_mat.strides[1]    = 1;

  res_mat.data          = A_f16_sdkl_scenario_3;
  res_mat.ndims         = 2;
  res_mat.dims[0]       = N_ROW;
  res_mat.dims[1]       = N_COL;
  res_mat.num_elements  = (N_ROW - 1) * res_stride + N_COL;
  res_mat.is_continuous = 0;
  res_mat.quantization  = SDKL_QUANT_NONE;
  res_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  res_mat.data_dtype    = SDKL_DTYPE_FP16;
  res_mat.data_offset   = 0;
  res_mat.strides[0]    = res_stride;
  res_mat.strides[1]    = 1;

  gettimeofday(&start, NULL);

  SDKL_CHECK(sdkl_mm_tensor(platform_npu, &res_mat, &left_mat, &right_mat));

  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 3, NPU runs %-.5lf s\n", time_reference);
}

/*!
  @brief
  Scenario 4: Matrix multiplication using SDKL NPU with Float16 precision and HMX-optimized layout.

  This scenario performs matrix multiplication using the SDKL NPU with:
  - FP16 weights in pre-transposed HMX layout (`SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX`)
  - FP16 activations in HMX layout (`SDKL_LAYOUT_2D_ROW_MAJOR_ACTIVATION_HMX`)
  - FP16 output in HMX layout

  The output tensor is also HMX-arranged. To compare it with a reference result
  generated using standard C code, the output must be converted to row-major layout.

  This function sets up three tensors for NPU-based matrix multiplication:
  - `right_mat`: FP16 weights, pre-transposed and arranged for HMX layout.
  - `left_mat`: FP16 input matrix, arranged for HMX layout.
  - `res_mat`: FP16 output matrix, arranged for HMX layout.

  The function configures tensor metadata including dimensions, strides, quantization type,
  and data layout, then invokes `sdkl_mm_tensor()` to perform the multiplication.

  Execution time is measured using `gettimeofday()` and printed to stdout.

  @see sdkl_mm_tensor
 */

_Float16* A_f16_ref_scenario_4  = NULL; /* Reference output */
_Float16* A_f16_sdkl_scenario_4 = NULL; /* SDKL output */

/*!
  @brief
  Reference code for matrix multiplication A = X * W^T
*/
__attribute__((noinline)) void matmul_ref_scenario_4(
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  _Float16* A,       // A[n_row][n_col]
  const _Float16* X, // X[n_row][n_inner]
  const _Float16* W  // W[n_col][n_inner]
) {
  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col; j++) {
      float acc = 0.0f;
      for (size_t k = 0; k < n_inner; k++) {
        acc += (float)X[i * n_inner + k] * (float)W[j * n_inner + k];
      }
      A[i * n_col + j] = (_Float16)acc;
    }
  }
}

/*!
  @brief
  NPU-offload, tensor-based code for matrix multiplication A = X * W^T
*/
void matmul_sdkl_scenario_4() {
  right_mat.data          = (void*)W_f16_npu;
  right_mat.ndims         = 2;
  right_mat.dims[0]       = N_INNER;
  right_mat.dims[1]       = N_COL;
  right_mat.num_elements  = N_INNER * N_COL + weight_offset;
  right_mat.is_continuous = 1;
  right_mat.quantization  = SDKL_QUANT_NONE;
  right_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX;
  right_mat.data_dtype    = SDKL_DTYPE_FP16;
  right_mat.data_offset   = 0;
  right_mat.strides[0]    = N_COL;
  right_mat.strides[1]    = 1;
  right_mat.data_offset   = weight_offset;

  left_mat.data          = (void*)X_f16_npu;
  left_mat.ndims         = 2;
  left_mat.dims[0]       = N_ROW;
  left_mat.dims[1]       = N_INNER;
  left_mat.num_elements  = N_ROW * N_INNER + x_offset;
  left_mat.is_continuous = 1;
  left_mat.quantization  = SDKL_QUANT_NONE;
  left_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR_ACTIVATION_HMX;
  left_mat.data_dtype    = SDKL_DTYPE_FP16;
  left_mat.data_offset   = x_offset;
  left_mat.strides[0]    = N_INNER;
  left_mat.strides[1]    = 1;

  res_mat.data          = A_f16_sdkl_scenario_4;
  res_mat.ndims         = 2;
  res_mat.dims[0]       = N_ROW;
  res_mat.dims[1]       = N_COL;
  res_mat.num_elements  = N_ROW * N_COL;
  res_mat.is_continuous = 1;
  res_mat.quantization  = SDKL_QUANT_NONE;
  res_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR_ACTIVATION_HMX;
  res_mat.data_dtype    = SDKL_DTYPE_FP16;
  res_mat.data_offset   = 0;
  res_mat.strides[0]    = N_COL;
  res_mat.strides[1]    = 1;

  gettimeofday(&start, NULL);

  SDKL_CHECK(sdkl_mm_tensor(platform_npu, &res_mat, &left_mat, &right_mat));

  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 4, NPU runs %-.5lf s\n", time_reference);
}

/*!
  @brief
  Scenario 5: Matrix multiplication using SDKL with FP16 inputs and continuous FP16 output layout.

  This scenario performs matrix multiplication on both NPU and CPU platforms using:
  - FP16 weights in standard row-major layout (`SDKL_LAYOUT_2D_ROW_MAJOR`)
  - FP16 activations in standard row-major layout (`SDKL_LAYOUT_2D_ROW_MAJOR`)
  - FP16 output in continuous row-major layout with offset

  This function sets up three tensors for matrix multiplication on both NPU and CPU platforms:
  - `right_mat`: FP16 weights, stored in standard row-major layout.
  - `left_mat`: FP16 input matrix, unquantized and stored in standard row-major layout.
  - `res_mat`: FP16 output matrix for NPU execution with continuous layout and data offset.
  - `res_cpu_mat`: FP16 output matrix for CPU execution with the same layout as NPU output.

  After executing the multiplication on the NPU, the same operation is repeated on the CPU using a separate output
  tensor. This allows direct performance comparison between hardware-accelerated and software-based execution paths
  using identical input data and tensor configurations.

  The function configures tensor metadata including dimensions, strides, quantization type,
  and data layout, then invokes `sdkl_mm_tensor()` for both platforms.

  Execution time is measured using `gettimeofday()` and printed to stdout for both NPU and CPU runs.

  @see sdkl_mm_tensor
*/

_Float16* A_f16_cpu_scenario_5  = NULL; /* SDKL CPU output */
_Float16* A_f16_sdkl_scenario_5 = NULL; /* SDKL NPU output */

/*!
  @brief
  NPU-offload and CPU, tensor-based code for matrix multiplication A = X * W^T
*/
void matmul_sdkl_scenario_5() {
  right_mat.data          = (void*)W_f16;
  right_mat.ndims         = 2;
  right_mat.dims[0]       = N_INNER;
  right_mat.dims[1]       = N_COL;
  right_mat.num_elements  = N_INNER * N_COL + weight_offset;
  right_mat.is_continuous = 1;
  right_mat.quantization  = SDKL_QUANT_NONE;
  right_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  right_mat.data_dtype    = SDKL_DTYPE_FP16;
  right_mat.data_offset   = 0;
  right_mat.strides[0]    = N_COL;
  right_mat.strides[1]    = 1;
  right_mat.data_offset   = weight_offset;

  left_mat.data          = (void*)X_f16; // Same weights matrix for NPU and CPU
  left_mat.ndims         = 2;
  left_mat.dims[0]       = N_ROW;
  left_mat.dims[1]       = N_INNER;
  left_mat.num_elements  = N_ROW * N_INNER + x_offset;
  left_mat.is_continuous = 1;
  left_mat.quantization  = SDKL_QUANT_NONE;
  left_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  left_mat.data_dtype    = SDKL_DTYPE_FP16;
  left_mat.data_offset   = x_offset;
  left_mat.strides[0]    = N_INNER;
  left_mat.strides[1]    = 1;

  res_mat.data          = A_f16_sdkl_scenario_5;
  res_mat.ndims         = 2;
  res_mat.dims[0]       = N_ROW;
  res_mat.dims[1]       = N_COL;
  res_mat.num_elements  = N_ROW * N_COL + res_offset;
  res_mat.is_continuous = 1;
  res_mat.quantization  = SDKL_QUANT_NONE;
  res_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  res_mat.data_dtype    = SDKL_DTYPE_FP16;
  res_mat.data_offset   = res_offset;
  res_mat.strides[0]    = N_COL;
  res_mat.strides[1]    = 1;

  gettimeofday(&start, NULL);

  /* NPU offload of matrix multiplication */
  SDKL_CHECK(sdkl_mm_tensor(platform_npu, &res_mat, &left_mat, &right_mat));

  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 5, NPU runs %-.5lf s\n", time_reference);

  res_cpu_mat.data          = A_f16_cpu_scenario_5;
  res_cpu_mat.ndims         = 2;
  res_cpu_mat.dims[0]       = N_ROW;
  res_cpu_mat.dims[1]       = N_COL;
  res_cpu_mat.num_elements  = N_ROW * N_COL + res_offset;
  res_cpu_mat.is_continuous = 1;
  res_cpu_mat.quantization  = SDKL_QUANT_NONE;
  res_cpu_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  res_cpu_mat.data_dtype    = SDKL_DTYPE_FP16;
  res_cpu_mat.data_offset   = res_offset;
  res_cpu_mat.strides[0]    = N_COL;
  res_cpu_mat.strides[1]    = 1;

  gettimeofday(&start, NULL);

  /* CPU Tensor matrix multiplication */
  SDKL_CHECK(sdkl_mm_tensor(platform_cpu, &res_cpu_mat, &left_mat, &right_mat));

  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 5, CPU Tensor single thread runs %-.5lf s\n", time_reference);
}

/*!
  @brief
  Scenario 6: Matrix multiplication using SDKL NPU with quantized inputs and 4-bit weights.

  This scenario performs matrix multiplication using the SDKL NPU with:
  - 4-bit weights in pre-transposed HMX layout (`SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX`)
  - 8-bit unsigned activations in standard row-major layout (`SDKL_LAYOUT_2D_ROW_MAJOR`)
  - 32-bit signed integer output in standard row-major layout

  This function sets up three tensors for NPU-based matrix multiplication:
  - `right_mat`: 4-bit weights, pre-transposed and arranged in HMX-optimized layout with 32-byte alignment padding.
  - `left_mat`: 8-bit unsigned input matrix, stored in standard row-major layout.
  - `res_mat`: 32-bit signed integer output matrix in continuous row-major layout.

  The weights require special handling with 32-byte alignment padding for both dimensions:
  - `n_col_32 = (N_COL + 31) & ~31` - Column dimension padded to 32-byte boundary
  - `n_inner_32 = (N_INNER + 31) & ~31` - Inner dimension padded to 32-byte boundary

  The 4-bit weights are packed (2 weights per byte) and converted from row-major to HMX layout
  using `sdkl_cpu_rm_to_wh_i4()` before NPU execution.

  The function configures tensor metadata including dimensions, strides, quantization type,
  and data layout, then invokes `sdkl_mm_tensor()` to perform the multiplication.

  Execution time is measured using `gettimeofday()` and printed to stdout.

  @see sdkl_mm_tensor
*/

int32_t* A_i32_ref_scenario_6  = NULL; /* Reference output */
int32_t* A_i32_sdkl_scenario_6 = NULL; /* SDKL output */

/*!
  @brief
  Reference code for matrix multiplication A = X * W^T
*/
__attribute__((noinline)) void matmul_ref_scenario_6(
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  int32_t* A,       // A[n_row][n_col]
  const uint8_t* X, // X[n_row][n_inner]
  const int8_t* W   // W[n_col][n_inner]
) {
  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col; j++) {
      int32_t acc = 0;
      for (size_t k = 0; k < n_inner; k++) {
        acc += (int32_t)X[i * n_inner + k] * (int32_t)W[j * n_inner + k];
      }
      A[i * n_col + j] = acc;
    }
  }
}

/*!
  @brief
  NPU-offload, tensor-based code for matrix multiplication A = X * W^T
*/
void matmul_sdkl_scenario_6() {
  size_t n_col_32   = (N_COL + 31) & ~31;
  size_t n_inner_32 = (N_INNER + 31) & ~31;

  right_mat.data          = (void*)W_ui8_npu_scenario_6;
  right_mat.ndims         = 2;
  right_mat.dims[0]       = n_inner_32;            // Use padded dimensions
  right_mat.dims[1]       = n_col_32;              // Use padded dimensions
  right_mat.num_elements  = n_inner_32 * n_col_32; // Total elements in padded tensor
  right_mat.is_continuous = 1;
  right_mat.quantization  = SDKL_QUANT_NONE;
  right_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX;
  right_mat.data_dtype    = SDKL_DTYPE_I4;
  right_mat.data_offset   = 0;
  right_mat.strides[0]    = n_col_32; // Use padded stride
  right_mat.strides[1]    = 1;

  left_mat.data          = (void*)X_ui8;
  left_mat.ndims         = 2;
  left_mat.dims[0]       = N_ROW;
  left_mat.dims[1]       = N_INNER;
  left_mat.num_elements  = N_ROW * N_INNER + x_offset;
  left_mat.is_continuous = 1;
  left_mat.quantization  = SDKL_QUANT_NONE;
  left_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  left_mat.data_dtype    = SDKL_DTYPE_U8;
  left_mat.data_offset   = x_offset;
  left_mat.strides[0]    = N_INNER;
  left_mat.strides[1]    = 1;

  res_mat.data          = A_i32_sdkl_scenario_6;
  res_mat.ndims         = 2;
  res_mat.dims[0]       = N_ROW;
  res_mat.dims[1]       = N_COL;
  res_mat.num_elements  = N_ROW * N_COL;
  res_mat.is_continuous = 1;
  res_mat.quantization  = SDKL_QUANT_NONE;
  res_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  res_mat.data_dtype    = SDKL_DTYPE_I32;
  res_mat.data_offset   = 0;
  res_mat.strides[0]    = N_COL;
  res_mat.strides[1]    = 1;

  gettimeofday(&start, NULL);

  SDKL_CHECK(sdkl_mm_tensor(platform_npu, &res_mat, &left_mat, &right_mat));

  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 6, NPU runs %-.5lf s\n", time_reference);
}

/*!
  @brief
  Scenario 7: Matrix multiplication using SDKL NPU with quantized inputs and 8-bit weights.

  This scenario performs matrix multiplication using the SDKL NPU with:
  - 8-bit signed weights in pre-transposed HMX layout (`SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX`)
  - 8-bit unsigned activations in standard row-major layout (`SDKL_LAYOUT_2D_ROW_MAJOR`)
  - 32-bit signed integer output in standard row-major layout

  This function sets up three tensors for NPU-based matrix multiplication:
  - `right_mat`: 8-bit signed weights, pre-transposed and arranged in HMX-optimized layout.
  - `left_mat`: 8-bit unsigned input matrix, stored in standard row-major layout.
  - `res_mat`: 32-bit signed integer output matrix in continuous row-major layout.

  Unlike Scenario 6, this scenario uses 8-bit weights (not 4-bit), so no special padding
  or packing is required. The weights are converted from row-major to HMX layout
  using `sdkl_cpu_rm_to_wh_i8_inplace()` before NPU execution.

  The function configures tensor metadata including dimensions, strides, quantization type,
  and data layout, then invokes `sdkl_mm_tensor()` to perform the multiplication.

  Execution time is measured using `gettimeofday()` and printed to stdout.

  @see sdkl_mm_tensor
*/

int32_t* A_i32_ref_scenario_7  = NULL; /* Reference output */
int32_t* A_i32_sdkl_scenario_7 = NULL; /* SDKL output */

/*!
  @brief
  Reference code for matrix multiplication A = X * W^T
*/
__attribute__((noinline)) void matmul_ref_scenario_7(
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  int32_t* A,       // A[n_row][n_col]
  const uint8_t* X, // X[n_row][n_inner]
  const int8_t* W   // W[n_col][n_inner]
) {
  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col; j++) {
      int32_t acc = 0;
      for (size_t k = 0; k < n_inner; k++) {
        acc += (int32_t)X[i * n_inner + k] * (int32_t)W[j * n_inner + k];
      }
      A[i * n_col + j] = acc;
    }
  }
}

/*!
  @brief
  NPU-offload, tensor-based code for matrix multiplication A = X * W^T
*/
void matmul_sdkl_scenario_7() {
  right_mat.data          = (void*)W_i8_npu_scenario_7;
  right_mat.ndims         = 2;
  right_mat.dims[0]       = N_INNER;
  right_mat.dims[1]       = N_COL;
  right_mat.num_elements  = N_INNER * N_COL;
  right_mat.is_continuous = 1;
  right_mat.quantization  = SDKL_QUANT_NONE;
  right_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX;
  right_mat.data_dtype    = SDKL_DTYPE_I8;
  right_mat.data_offset   = 0;
  right_mat.strides[0]    = N_COL;
  right_mat.strides[1]    = 1;

  left_mat.data          = (void*)X_ui8;
  left_mat.ndims         = 2;
  left_mat.dims[0]       = N_ROW;
  left_mat.dims[1]       = N_INNER;
  left_mat.num_elements  = N_ROW * N_INNER + x_offset;
  left_mat.is_continuous = 1;
  left_mat.quantization  = SDKL_QUANT_NONE;
  left_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  left_mat.data_dtype    = SDKL_DTYPE_U8;
  left_mat.data_offset   = x_offset;
  left_mat.strides[0]    = N_INNER;
  left_mat.strides[1]    = 1;

  res_mat.data          = A_i32_sdkl_scenario_7;
  res_mat.ndims         = 2;
  res_mat.dims[0]       = N_ROW;
  res_mat.dims[1]       = N_COL;
  res_mat.num_elements  = N_ROW * N_COL;
  res_mat.is_continuous = 1;
  res_mat.quantization  = SDKL_QUANT_NONE;
  res_mat.layout        = SDKL_LAYOUT_2D_ROW_MAJOR;
  res_mat.data_dtype    = SDKL_DTYPE_I32;
  res_mat.data_offset   = 0;
  res_mat.strides[0]    = N_COL;
  res_mat.strides[1]    = 1;

  gettimeofday(&start, NULL);

  SDKL_CHECK(sdkl_mm_tensor(platform_npu, &res_mat, &left_mat, &right_mat));

  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 7, NPU runs %-.5lf s\n", time_reference);
}

/*-------------- MAIN ------------------------------------------------------------------------------------*/
int main() {
  int res           = true;
  size_t X_f32_size = (N_ROW * N_INNER + x_offset) * sizeof(*X_f32);
  size_t X_f16_size = (N_ROW * N_INNER + x_offset) * sizeof(*X_f16);
  size_t W_f16_size = (N_COL * N_INNER + weight_offset) * sizeof(*W_f16);
  size_t X_ui8_size = (N_ROW * N_INNER + x_offset) * sizeof(*X_ui8);
  size_t W_i8_size  = (N_COL * N_INNER + weight_offset) * sizeof(*W_i8);

  // Initialize SDKL NPU
  SDKL_CHECK(sdkl_npu_initialize(platform_npu, NULL, NULL));

  SDKL_CHECK(sdkl_npu_get_version(platform_npu, version));

  printf("SDKL Version: %s\n", version);

  X_f32     = malloc(X_f32_size);
  X_f16     = malloc(X_f16_size);
  X_f16_npu = malloc(X_f16_size);
  W_f16     = malloc(W_f16_size);
  X_ui8     = malloc(X_ui8_size);
  W_i8      = malloc(W_i8_size);

  SDKL_CHECK(sdkl_npu_alloc(W_f16_size, (void**)&W_f16_npu));

  // Initialization by random values
  srand(42); //(unsigned int)time(NULL));
  printf("SDKL Test Start:\n");

  for (size_t i = 0; i < N_ROW; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      X_f32[i * N_INNER + j + x_offset]     = ((float)1.0f) * ((float)rand() / (float)RAND_MAX);
      X_f16[i * N_INNER + j + x_offset]     = (_Float16)X_f32[i * N_INNER + j + x_offset];
      X_f16_npu[i * N_INNER + j + x_offset] = X_f16[i * N_INNER + j + x_offset];
      X_ui8[i * N_INNER + j + x_offset]     = (uint8_t)(rand() % 127);
      rand();
    }
  }
  for (size_t i = 0; i < N_COL; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      W_f16[i * N_INNER + j + weight_offset]     = ((float)1.0f) * ((float)rand() / (float)RAND_MAX);
      W_f16_npu[i * N_INNER + j + weight_offset] = W_f16[i * N_INNER + j + weight_offset];
      rand();
      W_i8[i * N_INNER + j + weight_offset] = (int8_t)(rand() % 16) - 8;
    }
  }

  /* -------  SCENARIO 1 ------*/
  size_t A_f32_size_scenario_1 = N_ROW * N_COL * sizeof(*A_f32_ref_scenario_1);

  A_f32_ref_scenario_1  = malloc(A_f32_size_scenario_1);
  A_f32_sdkl_scenario_1 = malloc(A_f32_size_scenario_1);

  memset(A_f32_sdkl_scenario_1, 0, A_f32_size_scenario_1);
  memset(A_f32_ref_scenario_1, 0, A_f32_size_scenario_1);

  // Run and profile reference C code
  gettimeofday(&start, NULL);
  matmul_ref_scenario_1(N_ROW, N_COL, N_INNER, A_f32_ref_scenario_1, &X_f32[x_offset], &W_f16[weight_offset]);
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("Scenario 1, CPU single thread runs %-.5lf s\n", time_reference);

  // Perform Weights SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX layout. Performed in advance
  gettimeofday(&start, NULL);
  SDKL_CHECK(sdkl_cpu_rm_to_wh_f16_inplace(N_COL, N_INNER, &W_f16_npu[weight_offset]));
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 1, Weights data-layout runs %-.5lf s\n", time_reference);

  matmul_sdkl_scenario_1();

  // Check if the result differs from the reference
  res &= sdkl_vector_check_f32(N_ROW * N_COL, A_f32_ref_scenario_1, A_f32_sdkl_scenario_1);

  // Cleanup scenario 1 arrays
  if (A_f32_ref_scenario_1)
    free(A_f32_ref_scenario_1);
  if (A_f32_sdkl_scenario_1)
    free(A_f32_sdkl_scenario_1);

  /* -------  SCENARIO 2 ------*/
  size_t A_f32_size_scenario_2 = ((N_ROW - 1) * res_stride + N_COL) * sizeof(*A_f32_ref_scenario_2);

  A_f32_ref_scenario_2  = malloc(A_f32_size_scenario_2);
  A_f32_sdkl_scenario_2 = malloc(A_f32_size_scenario_2);

  memset(A_f32_sdkl_scenario_2, 0, A_f32_size_scenario_2);
  memset(A_f32_ref_scenario_2, 0, A_f32_size_scenario_2);

  // Run and profile reference C code
  gettimeofday(&start, NULL);
  matmul_ref_scenario_2(
    N_ROW, N_COL, N_INNER, res_stride, A_f32_ref_scenario_2, &X_f32[x_offset], &W_f16[weight_offset]
  );
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("Scenario 2, CPU single thread runs %-.5lf s\n", time_reference);

  matmul_sdkl_scenario_2();

  // Check if the result differs from the reference
  res &= sdkl_vector_check_f32(A_f32_size_scenario_2 / sizeof(float), A_f32_ref_scenario_2, A_f32_sdkl_scenario_2);

  // Cleanup scenario 2 arrays
  if (A_f32_ref_scenario_2)
    free(A_f32_ref_scenario_2);
  if (A_f32_sdkl_scenario_2)
    free(A_f32_sdkl_scenario_2);

  /* -------  SCENARIO 3 ------*/
  size_t A_f16_size_scenario_3 = ((N_ROW - 1) * res_stride + N_COL) * sizeof(*A_f16_ref_scenario_3);

  A_f16_ref_scenario_3  = malloc(A_f16_size_scenario_3);
  A_f16_sdkl_scenario_3 = malloc(A_f16_size_scenario_3);

  memset(A_f16_sdkl_scenario_3, 0, A_f16_size_scenario_3);
  memset(A_f16_ref_scenario_3, 0, A_f16_size_scenario_3);

  // Run and profile reference C code
  gettimeofday(&start, NULL);
  matmul_ref_scenario_3(
    N_ROW, N_COL, N_INNER, res_stride, A_f16_ref_scenario_3, &X_f16[x_offset], &W_f16[weight_offset]
  );
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("Scenario 3, CPU single thread runs %-.5lf s\n", time_reference);

  matmul_sdkl_scenario_3();

  // Check if the result differs from the reference
  res &= sdkl_vector_check_f16(A_f16_size_scenario_3 / sizeof(_Float16), A_f16_ref_scenario_3, A_f16_sdkl_scenario_3);

  // Cleanup scenario 3 arrays
  if (A_f16_ref_scenario_3)
    free(A_f16_ref_scenario_3);
  if (A_f16_sdkl_scenario_3)
    free(A_f16_sdkl_scenario_3);

  /* -------  SCENARIO 4 ------*/
  size_t A_f16_size_scenario_4 = ((N_ROW - 1) * res_stride + N_COL) * sizeof(*A_f16_ref_scenario_4);

  A_f16_ref_scenario_4  = malloc(A_f16_size_scenario_4);
  A_f16_sdkl_scenario_4 = malloc(A_f16_size_scenario_4);

  memset(A_f16_sdkl_scenario_4, 0, A_f16_size_scenario_4);
  memset(A_f16_ref_scenario_4, 0, A_f16_size_scenario_4);

  // Run and profile reference C code
  gettimeofday(&start, NULL);
  matmul_ref_scenario_4(N_ROW, N_COL, N_INNER, A_f16_ref_scenario_4, &X_f16[x_offset], &W_f16[weight_offset]);
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("Scenario 4, CPU single thread runs %-.5lf s\n", time_reference);

  // Perform Activation layout
  gettimeofday(&start, NULL);
  sdkl_cpu_rm_to_ah_f16_inplace(N_ROW, N_INNER, &X_f16_npu[x_offset]);
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 4, Activation data-layout runs %-.5lf s\n", time_reference);

  matmul_sdkl_scenario_4();

  // Perform Result layout. Required to enable comparison against a reference result that uses row-major layout
  gettimeofday(&start, NULL);
  sdkl_cpu_ah_to_rm_f16_inplace(N_ROW, N_COL, A_f16_sdkl_scenario_4);
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);

  printf("Scenario 4, Result data-layout runs %-.5lf s\n", time_reference);

  // Check if the result differs from the reference
  res &= sdkl_vector_check_f16(A_f16_size_scenario_4 / sizeof(_Float16), A_f16_ref_scenario_4, A_f16_sdkl_scenario_4);

  // Cleanup scenario 4 arrays
  if (A_f16_ref_scenario_4)
    free(A_f16_ref_scenario_4);
  if (A_f16_sdkl_scenario_4)
    free(A_f16_sdkl_scenario_4);

  /* -------  SCENARIO 5 ------*/
  size_t A_f16_size_scenario_5 = ((N_ROW - 1) * res_stride + N_COL + res_offset) * sizeof(*A_f16_cpu_scenario_5);

  A_f16_cpu_scenario_5  = malloc(A_f16_size_scenario_5);
  A_f16_sdkl_scenario_5 = malloc(A_f16_size_scenario_5);

  memset(A_f16_sdkl_scenario_5, 0, A_f16_size_scenario_5);
  memset(A_f16_cpu_scenario_5, 0, A_f16_size_scenario_5);

  matmul_sdkl_scenario_5();

  // Check if the result differs from the reference
  res &= sdkl_vector_check_f16(A_f16_size_scenario_5 / sizeof(_Float16), A_f16_cpu_scenario_5, A_f16_sdkl_scenario_5);

  // Cleanup scenario 5 arrays
  if (A_f16_cpu_scenario_5)
    free(A_f16_cpu_scenario_5);
  if (A_f16_sdkl_scenario_5)
    free(A_f16_sdkl_scenario_5);

  /* -------  SCENARIO 6 ------*/
  size_t A_i32_size_scenario_6 = N_ROW * N_COL * sizeof(int32_t);

  A_i32_ref_scenario_6  = malloc(A_i32_size_scenario_6);
  A_i32_sdkl_scenario_6 = malloc(A_i32_size_scenario_6);

  memset(A_i32_sdkl_scenario_6, 0, A_i32_size_scenario_6);
  memset(A_i32_ref_scenario_6, 0, A_i32_size_scenario_6);

  // Run and profile reference C code
  gettimeofday(&start, NULL);
  matmul_ref_scenario_6(N_ROW, N_COL, N_INNER, A_i32_ref_scenario_6, &X_ui8[x_offset], &W_i8[weight_offset]);
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("Scenario 6, CPU single thread runs %-.5lf s\n", time_reference);

  // Perform Weights layout conversion
  size_t n_col_32   = (N_COL + 31) & ~31;
  size_t n_inner_32 = (N_INNER + 31) & ~31;
  size_t W_i4_size  = ((n_col_32 * n_inner_32) % 2 ? (n_col_32 * n_inner_32 + 1) : (n_col_32 * n_inner_32)) / 2;

  SDKL_CHECK(sdkl_npu_alloc(W_i4_size, (void**)&W_ui8_npu_scenario_6));

  gettimeofday(&start, NULL);
  SDKL_CHECK(sdkl_cpu_rm_to_wh_i4(W_ui8_npu_scenario_6, &W_i8[weight_offset], n_inner_32, n_col_32));
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("Scenario 6, Weights data-layout runs %-.5lf s\n", time_reference);

  // Call NPU matrix multiplication using function like other scenarios
  matmul_sdkl_scenario_6();

  // Check if the result differs from the reference
  bool scenario_6_result = true;
  for (size_t i = 0; i < N_ROW; i++) {
    for (size_t j = 0; j < N_COL; j++) {
      if (A_i32_ref_scenario_6[j + N_COL * i] != A_i32_sdkl_scenario_6[j + N_COL * i]) {
        scenario_6_result = false;
        break;
      }
    }
    if (!scenario_6_result)
      break;
  }
  res &= scenario_6_result;

  // Cleanup scenario 6 arrays
  if (A_i32_ref_scenario_6)
    free(A_i32_ref_scenario_6);
  if (A_i32_sdkl_scenario_6)
    free(A_i32_sdkl_scenario_6);
  if (W_ui8_npu_scenario_6)
    sdkl_npu_free(W_ui8_npu_scenario_6);

  /* -------  SCENARIO 7 ------*/
  size_t A_i32_size_scenario_7 = N_ROW * N_COL * sizeof(int32_t);

  A_i32_ref_scenario_7  = malloc(A_i32_size_scenario_7);
  A_i32_sdkl_scenario_7 = malloc(A_i32_size_scenario_7);

  memset(A_i32_sdkl_scenario_7, 0, A_i32_size_scenario_7);
  memset(A_i32_ref_scenario_7, 0, A_i32_size_scenario_7);

  // Run and profile reference C code
  gettimeofday(&start, NULL);
  matmul_ref_scenario_7(N_ROW, N_COL, N_INNER, A_i32_ref_scenario_7, &X_ui8[x_offset], &W_i8[weight_offset]);
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("Scenario 7, CPU single thread runs %-.5lf s\n", time_reference);

  // Perform Weights layout conversion
  size_t W_i8_size_scenario_7 = N_COL * N_INNER * sizeof(int8_t);

  SDKL_CHECK(sdkl_npu_alloc(W_i8_size_scenario_7, (void**)&W_i8_npu_scenario_7));

  // Copy weight values to shared buffer
  for (size_t i = 0; i < N_COL; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      W_i8_npu_scenario_7[i * N_INNER + j] = W_i8[i * N_INNER + j + weight_offset];
    }
  }

  gettimeofday(&start, NULL);
  SDKL_CHECK(sdkl_cpu_rm_to_wh_i8_inplace(N_COL, N_INNER, W_i8_npu_scenario_7));
  gettimeofday(&end, NULL);

  time_reference = elapsed(start, end);
  printf("Scenario 7, Weights data-layout runs %-.5lf s\n", time_reference);

  // Call NPU matrix multiplication using function like other scenarios
  matmul_sdkl_scenario_7();

  // Check if the result differs from the reference
  bool scenario_7_result = true;
  for (size_t i = 0; i < N_ROW; i++) {
    for (size_t j = 0; j < N_COL; j++) {
      if (A_i32_ref_scenario_7[j + N_COL * i] != A_i32_sdkl_scenario_7[j + N_COL * i]) {
        scenario_7_result = false;
        break;
      }
    }
    if (!scenario_7_result)
      break;
  }
  res &= scenario_7_result;

  // Cleanup scenario 7 arrays
  if (A_i32_ref_scenario_7)
    free(A_i32_ref_scenario_7);
  if (A_i32_sdkl_scenario_7)
    free(A_i32_sdkl_scenario_7);
  if (W_i8_npu_scenario_7)
    sdkl_npu_free(W_i8_npu_scenario_7);

  if (res) {
    printf("Test Passed\n");
  } else {
    printf("Test Failed\n");
  }

  // Cleanup global arrays
  if (X_f32)
    free(X_f32);
  if (X_f16)
    free(X_f16);
  if (X_f16_npu)
    free(X_f16_npu);
  if (W_f16)
    free(W_f16);
  if (X_ui8)
    free(X_ui8);
  if (W_i8)
    free(W_i8);
  if (W_f16_npu)
    sdkl_npu_free(W_f16_npu);

  // Finalize & cleanup SDKL
  SDKL_CHECK(sdkl_npu_finalize(platform_npu));

  return res ? 0 : 1;
}
