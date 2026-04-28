// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

#include "AEEStdErr.h"
#include "remote.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hexkl_macro.h"
#include "hexkl_micro.h"

#define MAX_VAL (5U) // Random values are in [0, MAX_VAL]
#define N_ROW   (32U)
#define N_COL   (128U)
#define N_INNER (64U)

/*!
  @brief
  Compares HEXKL MICRO API result vs Standard C reference. Tolerates 0.1% error
*/
int hexkl_vector_check_f32(size_t size, _Float16* ref, _Float16* vec) {
  int res = AEE_SUCCESS;
  for (int32_t i = 0; i < size; i++) {
    float diff;
    float diff_0dot001percent = fabsf((float)ref[i] / (float)1000.0f);

    if (isnan((float)ref[i])) {
      res = AEE_EFAILED;
      printf(
        "[HEXKL_MACRO][ERROR] ISNAN ref[%ld] = %f vec[%ld] = %f\n", (long)i, (float)ref[i], (long)i, (float)vec[i]
      );
      break;
    }

    if (isinf((float)vec[i])) {
      res = AEE_EFAILED;
      printf(
        "[HEXKL_MACRO][ERROR] ISINF ref[%ld] = %f vec[%ld] = %f\n", (long)i, (float)ref[i], (long)i, (float)vec[i]
      );
      break;
    }
    diff = fabsf((float)ref[i] - (float)vec[i]);
    if ((diff > diff_0dot001percent) && (diff > 0.01)) {
      res = AEE_EFAILED;
      printf(
        "[HEXKL_MACRO][ERROR] ref[%ld] = %f vec[%ld] = %f, diff = %f, tolerated epsilon = %f\n",
        (long)i,
        (float)ref[i],
        (long)i,
        (float)vec[i],
        diff,
        diff_0dot001percent
      );
      break;
    }
  }
  return res;
}

/*!
 @brief
 Reference Standard C code of Matrix multiplication A = X * W^T
*/
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

char version[256];

int main() {
  int res                   = AEE_SUCCESS;
  int res2                  = AEE_SUCCESS;
  _Float16* A_f16_reference = NULL;
  _Float16* A_f16           = NULL;
  _Float16* X_f16           = NULL;
  _Float16* W_f16           = NULL;
  _Float16* W_f16_npu       = NULL;
  size_t A_f16_size         = N_ROW * N_COL * sizeof(*A_f16_reference);
  size_t X_f16_size         = N_ROW * N_INNER * sizeof(*X_f16);
  size_t W_f16_size         = N_COL * N_INNER * sizeof(*W_f16);

  printf("[HEXKL_MACRO] Test Start:\n");

  A_f16_reference = malloc(A_f16_size);
  A_f16           = malloc(A_f16_size);
  X_f16           = malloc(X_f16_size);
  W_f16           = malloc(W_f16_size);
  W_f16_npu       = malloc(W_f16_size);

  res = hexkl_macro_initialize();
  if (res != AEE_SUCCESS) {
    printf("[HEXKL_MACRO][ERROR] hexkl_macro_initialize failed\n");
    goto TEST_END;
  } else {
    printf("[HEXKL_MACRO] hexkl_macro_initialize OK\n");
  }

  hexkl_macro_get_version(version);
  if (res != AEE_SUCCESS) {
    printf("[HEXKL_MACRO][ERROR] Version access failed\n");
    goto TEST_END;
  } else {
    printf("[HEXKL_MACRO] Version is: %s\n", version);
  }

  res = hexkl_macro_lock_hmx();
  if (res != AEE_SUCCESS) {
    printf("[HEXKL_MACRO][ERROR] HMX Lock failed\n");
    goto TEST_END;
  } else {
    printf("[HEXKL_MACRO] HMX Lock OK\n");
  }

  // Initialization
  for (size_t i = 0; i < N_ROW; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      X_f16[i * N_INNER + j] = (_Float16)((float)1.0f) * ((float)((i % 5) + 0.067f));
    }
  }

  printf("[HEXKL_MACRO] X_f16 init done\n");

  for (size_t i = 0; i < N_COL; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      W_f16[i * N_INNER + j]     = (_Float16)((float)1.0f) * ((float)((i % 3) + 0.04906f));
      W_f16_npu[i * N_INNER + j] = W_f16[i * N_INNER + j];
    }
  }

  printf("[HEXKL_MACRO] W_f16 init done\n");

  for (size_t i = 0; i < N_ROW * N_COL; i++) {
    A_f16_reference[i] = (_Float16)0.0f;
    A_f16[i]           = (_Float16)0.0f;
  }

  printf("[HEXKL_MACRO] A_f16 init done\n");

  matmul(N_ROW, N_COL, N_INNER, A_f16_reference, X_f16, W_f16);

  printf("[HEXKL_MACRO] Standard C matmul done\n");

  // Perform Weights layout
  hexkl_macro_rm_to_wh_f16_inplace(N_COL, N_INNER, W_f16_npu);

  // Perform Activation layout
  hexkl_macro_rm_to_ah_f16_inplace(N_ROW, N_INNER, X_f16);

  res = hexkl_macro_mm_f16(N_ROW, N_COL, N_INNER, A_f16, X_f16, W_f16_npu);
  if (res != AEE_SUCCESS) {
    printf("[HEXKL_MACRO][ERROR] hexkl_macro_mm_f16 failed. error code: 0x%x\n", res);
    goto TEST_END;
  } else {
    printf("[HEXKL_MACRO] hexkl_macro_mm_f16 done\n");
  }

  // Perform Result layout
  hexkl_macro_ah_to_rm_f16_inplace(N_ROW, N_COL, A_f16);

  res = hexkl_vector_check_f32(N_ROW * N_COL, A_f16_reference, A_f16);
  if (res != AEE_SUCCESS) {
    printf("[HEXKL_MACRO][ERROR] HMX matmul error not within tolerance\n");
    goto TEST_END;
  }

TEST_END:
  res2 = hexkl_macro_unlock_hmx();
  if (res2 != AEE_SUCCESS) {
    res |= res2;
    printf("[HEXKL_MACRO][ERROR] HMX Unlock failed\n");
  } else {
    printf("[HEXKL_MACRO] HMX Unlock OK\n");
  }

  res2 = hexkl_macro_finalize();
  if (res2 != AEE_SUCCESS) {
    res |= res2;
    printf("[HEXKL_MACRO][ERROR] hexkl_macro_finalize failed\n");
  } else {
    printf("[HEXKL_MACRO] hexkl_macro_finalize OK\n");
  }

  if (A_f16_reference)
    free(A_f16_reference);
  if (A_f16)
    free(A_f16);
  if (X_f16)
    free(X_f16);
  if (W_f16)
    free(W_f16);
  if (W_f16_npu)
    free(W_f16_npu);

  if (res == AEE_SUCCESS) {
    printf("[HEXKL_MACRO] Test Passed\n");
  } else {
    printf("[HEXKL_MACRO] Test Failed\n");
  }

  return res;
}
