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

#include "hexkl_micro.h"

#define MAX_VAL (5U) // Random values are in [0, MAX_VAL]
#define N_ROW   (64U)
#define N_COL   (128U)
#define N_INNER (128U)

/**
  @def HEXKL_HMX_MAX_TILES_IN_ACTIVATION
  @brief Maximum number of activation tiles assumed in VTCM.

  This macro defines the assumed upper limit on the number of activation tiles
  that can be allocated in VTCM before weight tiles are placed. It represents
  a sample partitioning strategy for VTCM usage in HMX-based matrix multiplication.

  > **Note:** This is only an example configuration. The actual partitioning
  strategy should be determined by the NPU programmer based on application needs.
 */
#define HEXKL_HMX_MAX_TILES_IN_ACTIVATION (510U)

/*!
  @brief
  Compares HEXKL MICRO API result vs Standard C reference.
*/
int hexkl_vector_check_i32(size_t size, int32_t* ref, int32_t* vec) {
  int res = AEE_SUCCESS;
  for (int32_t i = 0; i < size; i++) {
    int32_t diff;

    diff = ref[i] - vec[i];
    if (diff) {
      res = AEE_EFAILED;
      printf("[HEXKL_MICRO][ERROR] ref[%ld] = %ld vec[%ld] = %ld\n", (long)i, (long)ref[i], (long)i, (long)vec[i]);
      break;
    }
  }
  return res;
}

/*!
  @brief
  Reference Standard C code
*/
static int matmul(
  uint32_t A_rows,
  uint32_t n_inner,
  uint32_t W_cols,
  int32_t* restrict matX,
  const uint8_t* restrict matA,
  const int8_t* restrict matW
) {
  int ret = AEE_SUCCESS;
  for (int row = 0; row < A_rows; row++) {
    for (int col = 0; col < W_cols; col++) {
      int32_t acc = 0;
      for (int it = 0; it < n_inner; it++) {
        acc += (uint32_t)matA[row * n_inner + it] * (int32_t)matW[it * W_cols + col];
      }
      matX[row * W_cols + col] = acc;
    }
  }

  return ret;
}

int hexkl_micro_matmul_u8i8_i32(
  uint8_t* vtcm_base,
  uint32_t vtcm_size,
  uint32_t A_rows,
  uint32_t n_inner,
  uint32_t W_cols,
  int32_t* matA,
  uint8_t* matX,
  int8_t* matW
) {
  int ret                 = AEE_SUCCESS;
  uint32_t X_rows         = A_rows;
  uint32_t X_cols         = W_cols;
  uint32_t A_cols         = n_inner;
  uint32_t row_tiles_in_A = A_cols / HEXKL_HMX_INT8_BLOCK_N_INNER;

  // Put HMX config at end of allocated VTCM
  uint32_t hmx_config_offset = vtcm_size - hexkl_micro_hmx_config_size();

  // Store accumulator reads well beyond source tiles and intermediate values
  uint32_t result_offset = hmx_config_offset - HEXKL_HMX_INT8_BLOCK_N_INNER * HEXKL_HMX_INT8_BLOCK_N_ROW * 4;

  uint32_t weight_offset =
    HEXKL_HMX_INT8_BLOCK_N_ROW * HEXKL_HMX_INT8_BLOCK_N_INNER * HEXKL_HMX_MAX_TILES_IN_ACTIVATION;

  if ((vtcm_size == 0) || (vtcm_size % HEXKL_HMX_ACTIVATION_ALIGNMENT != 0)) {
    printf("[HEXKL_MICRO][ERROR] Illegal VTCM size = 0x%x bytes", (int)vtcm_size);
    return AEE_ENOMEMORY;
  }

  hexkl_micro_hmx_setup_acc_read_int32(vtcm_base, hmx_config_offset);

  // Iterate through rows of X at tile height stride
  for (uint32_t row = 0; row < A_rows; row += HEXKL_HMX_INT8_BLOCK_N_ROW) {

    // Load one row of tiles from A. Store row starting at vtcm_base
    for (int i = 0; i < row_tiles_in_A; i++) {
      hexkl_micro_hmx_copy_submatrix_to_8b_activation(
        vtcm_base,
        /*out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
        /*input_matrix=*/matX,
        /*tile_row=*/row / HEXKL_HMX_INT8_BLOCK_N_ROW,
        /*tile_col=*/i,
        /*input_rows=*/A_rows,
        /*input_cols=*/A_cols
      );

      // No layout needed
    }
    // The tiles for one row of A are now loaded. These tiles occupy the first
    // (row_tiles_in_A * HEXKL_HMX_ACTIVATION_ALIGNMENT bytes) of memory from vtcm_base.

    // Iterate through columns of X at tile width stride
    uint32_t col = 0;
    for (; col < W_cols; col += HEXKL_HMX_INT8_BLOCK_N_COL) {

      hexkl_micro_hmx_acc_clear_int32();
      // Iterate through (one row of tiles in A) * (one col of tiles in W)
      for (int i = 0; i < row_tiles_in_A; i++) {
        // Copy a weight tile from a submatrix.
        // Each int8 tile of W is 32x32 = 1024 bytes
        hexkl_micro_hmx_rm_to_wh_i8(vtcm_base, weight_offset, matW, i, (col) / 32, W_cols);

        hexkl_micro_hmx_mm_u8i8(
          vtcm_base,
          /*activation_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
          /*weight_offset=*/weight_offset
        );
      }

      // Read 64x32 int32 accumulator
      hexkl_micro_hmx_acc_read_int32(
        vtcm_base,
        hmx_config_offset,
        /*out_offset=*/result_offset
      );
      // Copy to X
      hexkl_micro_hmx_copy_32b_to_submatrix(
        vtcm_base,
        /*in_offset=*/result_offset,
        /*output_matrix=*/matA,
        /*tile_row=*/row / HEXKL_HMX_INT8_BLOCK_N_ROW,
        /*tile_col=*/col / HEXKL_HMX_INT8_BLOCK_N_COL,
        /*output_rows=*/X_rows,
        /*output_cols=*/X_cols
      );
    }
  }

  return ret;
}

char version[256];

int main() {
  int res                  = AEE_SUCCESS;
  int res2                 = AEE_SUCCESS;
  int32_t* A_i32_reference = NULL;
  int32_t* A_i32           = NULL;
  uint8_t* X_u8            = NULL;
  int8_t* W_i8             = NULL;
  size_t A_i32_size        = N_ROW * N_COL * sizeof(*A_i32_reference);
  size_t X_u8_size         = N_ROW * N_INNER * sizeof(*X_u8);
  size_t W_i8_size         = N_COL * N_INNER * sizeof(*W_i8);
  uint8_t* vtcm_base       = NULL;
  uint32_t vtcm_size       = 0;
  int major                = 0;
  int minor                = 0;
  int patch                = 0;
  int hex_version          = 0;
  char version_prerel[HEXKL_PREREL_STR_LEN];

  printf("[HEXKL_MICRO] Test Start:\n");

  A_i32_reference = malloc(A_i32_size);
  A_i32           = malloc(A_i32_size);
  X_u8            = malloc(X_u8_size);
  W_i8            = malloc(W_i8_size);

  res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size);
  if (res != AEE_SUCCESS) {
    printf("[HEXKL_MICRO][ERROR] Init failed\n");
    goto TEST_END;
  } else {
    printf("[HEXKL_MICRO] VTCM base = 0x%p  VTCM size = %d bytes:\n", vtcm_base, (int)vtcm_size);
  }

  hexkl_micro_get_version(&major, &minor, &patch, version_prerel, &hex_version);
  if (res != AEE_SUCCESS) {
    printf("[HEXKL_MICRO][ERROR] Version access failed\n");
    goto TEST_END;
  } else {
    sprintf(version, "%d_%d_%d_%s_HEXAGON_V%d", major, minor, patch, version_prerel, hex_version);
    printf("[HEXKL_MICRO] Version is: %s\n", version);
  }

  res = hexkl_micro_hmx_lock();
  if (res != AEE_SUCCESS) {
    printf("[HEXKL_MICRO][ERROR] HMX Lock failed\n");
    goto TEST_END;
  } else {
    printf("[HEXKL_MICRO] HMX Lock OK\n");
  }

  // Initialization
  for (size_t i = 0; i < N_ROW; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      X_u8[i * N_INNER + j] = (uint8_t)((i * j + 2 + i) & 0xFF);
    }
  }

  printf("[HEXKL_MICRO] X_u8 init done\n");

  for (size_t i = 0; i < N_COL; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      W_i8[i * N_INNER + j] = (int8_t)((i + j * 3 - i / 4) & 0xFF);
    }
  }

  printf("[HEXKL_MICRO] W_i8 init done\n");

  for (size_t i = 0; i < N_ROW * N_COL; i++) {
    A_i32_reference[i] = 0;
    A_i32[i]           = 0;
  }

  printf("[HEXKL_MICRO] A_i32 init done\n");

  matmul(N_ROW, N_COL, N_INNER, A_i32_reference, X_u8, W_i8);

  printf("[HEXKL_MICRO] Standard C matmul done\n");

  hexkl_micro_matmul_u8i8_i32(vtcm_base, vtcm_size, N_ROW, N_COL, N_INNER, A_i32, X_u8, W_i8);

  printf("[HEXKL_MICRO] HMX matmul done\n");

  res = hexkl_vector_check_i32(N_ROW * N_COL, A_i32_reference, A_i32);
  if (res != AEE_SUCCESS) {
    printf("[HEXKL_MICRO][ERROR] HMX matmul not bit-exact\n");
    goto TEST_END;
  }

TEST_END:
  res2 = hexkl_micro_hmx_unlock();
  if (res2 != AEE_SUCCESS) {
    printf("[HEXKL_MICRO][ERROR] HMX Unlock failed\n");
    res |= res2;
  } else {
    printf("[HEXKL_MICRO] HMX Unlock OK\n");
  }

  if (A_i32_reference)
    free(A_i32_reference);
  if (A_i32)
    free(A_i32);
  if (X_u8)
    free(X_u8);
  if (W_i8)
    free(W_i8);

  if (res == AEE_SUCCESS) {
    printf("[HEXKL_MICRO] Test Passed\n");
  } else {
    printf("[HEXKL_MICRO] Test Failed\n");
  }

  return res;
}
