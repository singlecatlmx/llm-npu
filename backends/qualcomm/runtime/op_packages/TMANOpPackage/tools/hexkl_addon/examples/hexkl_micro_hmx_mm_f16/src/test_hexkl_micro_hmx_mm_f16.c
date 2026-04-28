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
#define N_ROW   (32U)
#define N_COL   (128U)
#define N_INNER (64U)

/*!
  @brief
  Compares HEXKL MICRO API result vs Standard C reference. Tolerates 0.1% error
*/
int hexkl_vector_check_f32(size_t size, float* ref, float* vec) {
  int res = AEE_SUCCESS;
  for (int32_t i = 0; i < size; i++) {
    float diff;
    float diff_0dot001percent = fabsf(ref[i] / (float)1000.0f);

    if (isnan((float)ref[i])) {
      res = AEE_EFAILED;
      printf(
        "[HEXKL_MICRO][ERROR] ISNAN ref[%ld] = %f vec[%ld] = %f\n", (long)i, (float)ref[i], (long)i, (float)vec[i]
      );
      break;
    }

    if (isinf((float)vec[i])) {
      res = AEE_EFAILED;
      printf(
        "[HEXKL_MICRO][ERROR] ISINF ref[%ld] = %f vec[%ld] = %f\n", (long)i, (float)ref[i], (long)i, (float)vec[i]
      );
      break;
    }
    diff = fabsf(ref[i] - vec[i]);
    if ((diff > diff_0dot001percent) && (diff > 0.01)) {
      res = AEE_EFAILED;
      printf(
        "[HEXKL_MICRO][ERROR] ref[%ld] = %f vec[%ld] = %f, diff = %f, tolerated epsilon = %f\n",
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
 Reference Standard C code
*/
static int matmul(
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  float* restrict outM,
  const _Float16* restrict inAct,
  const _Float16* restrict inW
) {
  int ret = AEE_SUCCESS;

  uint32_t N      = n_row;
  uint32_t K      = n_inner;
  uint32_t M      = n_col;
  uint32_t W_cols = n_col;
  uint32_t A_cols = n_inner;
  uint32_t X_cols = n_col;

  for (uint32_t row = 0; row < N; row++) {
    for (uint32_t col = 0; col < M; col++) {
      float acc = 0;
      for (uint32_t entry = 0; entry < K; entry++) {
        acc += (float)inAct[row * A_cols + entry] * inW[entry * W_cols + col];
      }
      outM[row * X_cols + col] = acc;
    }
  }

  return ret;
}

int hexkl_micro_matmul_f16f16_f32(
  uint8_t* vtcm_base,
  uint32_t vtcm_size,
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  float* restrict outM,
  const _Float16* restrict inAct,
  const _Float16* restrict inW
) {
  int ret = AEE_SUCCESS;

  if ((vtcm_size == 0) || (vtcm_size % HEXKL_HMX_ACTIVATION_ALIGNMENT != 0)) {
    printf("[HEXKL_MICRO][ERROR] Illegal VTCM size = 0x%x bytes", (int)vtcm_size);
    return AEE_ENOMEMORY;
  }

  uint32_t N              = n_row;
  uint32_t M              = n_col;
  uint32_t A_rows         = n_row;
  uint32_t A_cols         = n_inner;
  uint32_t X_cols         = n_col;
  uint32_t X_rows         = n_row;
  uint32_t row_tiles_in_A = (n_inner + (HEXKL_HMX_F16_BLOCK_N_INNER - 1)) / HEXKL_HMX_F16_BLOCK_N_INNER;
  uint32_t weight_offset  = HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A;
  const _Float16* matA    = (const _Float16*)inAct;
  const _Float16* matW    = (const _Float16*)inW;
  float* matX             = outM;

  // Put HMX config at end of allocated VTCM
  uint32_t hmx_config_offset = vtcm_size - hexkl_micro_hmx_config_size();
  hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, hmx_config_offset);

  // Iterate through rows of X at tile height stride
  for (uint32_t row = 0; row < N; row += HEXKL_HMX_F16_BLOCK_N_ROW) {
    // Load and layout one row of tiles from A. Store row starting at vtcm_base.
    for (int i = 0; i < row_tiles_in_A; i++) {
      // Each fp16 tile of A is 32x32 = 2048 bytes
      hexkl_micro_hmx_copy_submatrix_to_f16(
        vtcm_base,
        /*out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i),
        /*input_matrix=*/matA,
        /*tile_row=*/row / HEXKL_HMX_F16_BLOCK_N_ROW,
        /*tile_col=*/i,
        /*input_rows=*/A_rows,
        /*input_cols=*/A_cols
      );
      hexkl_micro_hmx_rm_to_ah_f16(
        vtcm_base,
        /*activation_out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
        /*flat_in_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i)
      );
    }
    // The tiles for one row of A are now loaded and laid out correctly
    // These tiles occupy the first (row_tiles_in_A * HEXKL_HMX_ACTIVATION_ALIGNMENT bytes) of memory from
    // vtcm_base.

    // Iterate through columns of X at tile width stride
    uint32_t col = 0;
    for (; col < M; col += 32) {
      hexkl_micro_hmx_acc_clear_f16();
      // Iterate through (one row of tiles in A) * (one col of tiles in W)
      for (int i = 0; i < row_tiles_in_A; i++) {

        hexkl_micro_hmx_rm_to_wh_f16(
          vtcm_base,
          /*weight_offset=*/weight_offset,
          matW,
          /*row_tile =*/i,
          /*col_tile =*/(col) / 32,
          /*wt_cols*/ n_col
        );

        hexkl_micro_hmx_mm_f16(
          vtcm_base,
          /*activation_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
          /*weight_offset=*/weight_offset
        );
      }
      // Read 32x32 fp16 accumulator
      hexkl_micro_hmx_acc_read_f16(
        vtcm_base,
        hmx_config_offset,
        /*out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1)
      );
      // Change layout to row major
      hexkl_micro_hmx_ah_to_rm_f16(
        vtcm_base,
        /*flat_out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
        /*activation_in_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1)
      );
      // Copy into X
      hexkl_micro_hmx_copy_f16_to_f32_submatrix(
        vtcm_base,
        /*in_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
        /*output_matrix=*/matX,
        /*tile_row=*/row / HEXKL_HMX_F16_BLOCK_N_ROW,
        /*tile_col=*/col / HEXKL_HMX_F16_BLOCK_N_COL,
        /*output_rows=*/X_rows,
        /*output_cols=*/X_cols
      );
    }
  }

  return ret;
}

char version[256];

int main() {
  int res                = AEE_SUCCESS;
  int res2               = AEE_SUCCESS;
  float* A_f32_reference = NULL;
  float* A_f32           = NULL;
  _Float16* X_f16        = NULL;
  _Float16* W_f16        = NULL;
  size_t A_f32_size      = N_ROW * N_COL * sizeof(*A_f32_reference);
  size_t X_f16_size      = N_ROW * N_INNER * sizeof(*X_f16);
  size_t W_f16_size      = N_COL * N_INNER * sizeof(*W_f16);
  uint8_t* vtcm_base     = NULL;
  uint32_t vtcm_size     = 0;
  int major              = 0;
  int minor              = 0;
  int patch              = 0;
  int hex_version        = 0;
  char version_prerel[HEXKL_PREREL_STR_LEN];

  printf("[HEXKL_MICRO] Test Start:\n");

  A_f32_reference = malloc(A_f32_size);
  A_f32           = malloc(A_f32_size);
  X_f16           = malloc(X_f16_size);
  W_f16           = malloc(W_f16_size);

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
      X_f16[i * N_INNER + j] = (_Float16)((float)1.0f) * ((float)((i % 5) + 0.067f));
    }
  }

  printf("[HEXKL_MICRO] X_f16 init done\n");

  for (size_t i = 0; i < N_COL; i++) {
    for (size_t j = 0; j < N_INNER; j++) {
      W_f16[i * N_INNER + j] = (_Float16)((float)1.0f) * ((float)((i % 3) + 0.04906f));
    }
  }

  printf("[HEXKL_MICRO] W_f16 init done\n");

  for (size_t i = 0; i < N_ROW * N_COL; i++) {
    A_f32_reference[i] = 0.0f;
    A_f32[i]           = 0.0f;
  }

  printf("[HEXKL_MICRO] A_f32 init done\n");

  matmul(N_ROW, N_COL, N_INNER, A_f32_reference, X_f16, W_f16);

  printf("[HEXKL_MICRO] Standard C matmul done\n");

  hexkl_micro_matmul_f16f16_f32(vtcm_base, vtcm_size, N_ROW, N_COL, N_INNER, A_f32, X_f16, W_f16);

  printf("[HEXKL_MICRO] HMX matmul done\n");

  res = hexkl_vector_check_f32(N_ROW * N_COL, A_f32_reference, A_f32);
  if (res != AEE_SUCCESS) {
    printf("[HEXKL_MICRO][ERROR] HMX matmul error not within tolerance\n");
    goto TEST_END;
  }

TEST_END:
  res2 = hexkl_micro_hmx_unlock();
  if (res2 != AEE_SUCCESS) {
    res |= res2;
    printf("[HEXKL_MICRO][ERROR] HMX Unlock failed\n");
  } else {
    printf("[HEXKL_MICRO] HMX Unlock OK\n");
  }

  if (A_f32_reference)
    free(A_f32_reference);
  if (A_f32)
    free(A_f32);
  if (X_f16)
    free(X_f16);
  if (W_f16)
    free(W_f16);

  if (res == AEE_SUCCESS) {
    printf("[HEXKL_MICRO] Test Passed\n");
  } else {
    printf("[HEXKL_MICRO] Test Failed\n");
  }

  return res;
}
