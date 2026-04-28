// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

#ifndef __HEXKL_MACRO_H__
#define __HEXKL_MACRO_H__

#include <stdint.h>

/*!
  @file hexkl_macro.h
  @brief Defines constants and functions in the HexKL NPU Macro API.
*/

/*!
  @defgroup HexKLNPUMacro HexKL NPU Macro API
  @brief NPU functions for matrix multiplication and layout transformation.
*/

#ifdef __cplusplus
extern "C" {
#endif
/*!
  @ingroup HexKLNPUMacro
  @defgroup NPUMacroConstants Constants
  @brief Defines constants for tensor sizes, memory alignment, and version string lengths.
*/

/*!
  @ingroup NPUMacroConstants
  @def HEXKL_N_ROW_BYTES_MAX
  @brief Maximum number of bytes per output row supported for matrix multiplication.

  This value defines the upper limit of row size (in bytes) for matrix
  multiplication operations on Hexagon architectures v73/v75/v79
  and reflects constraints imposed by the 32-bit address space on these
  architectures.

  This size is tuned to support a total memory footprint of approximately 2 GB.
*/
#define HEXKL_N_ROW_BYTES_MAX (3200U)

/*!
  @ingroup NPUMacroConstants
  @def HEXKL_N_COL_BYTES_MAX
  @brief Maximum number of bytes per output column supported for matrix multiplication.

  This constant sets the maximum column size (in bytes) for matrix
  multiplication on Hexagon v73/v75/v79 and reflects constraints imposed by the
  32-bit address space on these architectures.

  This size is tuned to support a total memory footprint of approximately 2 GB.
*/
#define HEXKL_N_COL_BYTES_MAX (10240U)

/*!
  @ingroup NPUMacroConstants
  @def HEXKL_N_INNER_BYTES_MAX
  @brief Maximum number of bytes for the inner dimension in matrix multiplication.

  This value represents the largest supported size (in bytes) for the
  inner dimension of matrix multiplication on Hexagon v73/v75/v79 and reflects
  constraints imposed by the 32-bit address space on these architectures.

  This size is tuned to support a total memory footprint of approximately 2 GB.
*/
#define HEXKL_N_INNER_BYTES_MAX (152060U)

/*!
  @note
  The functions declared below are intended for use exclusively by
  NPU/Hexagon programmers. These functions may rely on Hexagon-specific
  hardware features or optimizations and are not guaranteed to be
  portable across other architectures.

  The macro __hexagon__ is defined by the Hexagon compiler to indicate
  compilation for the Hexagon DSP architecture.
*/
#ifdef __hexagon__

/*!
  @ingroup HexKLNPUMacro
  @defgroup NPUMacroGeneral General Functions
  @brief Defines general functions
*/
/*!
  @ingroup NPUMacroGeneral
  @brief Initializes the HexKL macro engine for NPU operations.

  This function sets up internal resources required by the HexKL library
  to enable macro-level operations on the NPU. It must be called before
  invoking any other HexKL-related functions.

  @return
  - `AEE_SUCCESS` on successful initialization.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_ENOMEM`, etc.) in case of failure.

  @note
  This function should be called once, before all HexKL Macro function calls.
 */
int hexkl_macro_initialize(void);

/*!
  @ingroup NPUMacroGeneral
  @brief Deinitializes the HexKL library and releases macro engine resources.

  This function cleans up internal resources allocated during the initialization
  and usage of the HexKL library for macro-level NPU operations. It should be called
  when HexKL functionality is no longer required, typically during application shutdown.

  @return
  - `AEE_SUCCESS` on successful deinitialization.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADSTATE`, etc.) in case of failure.

  @note
  This function should be called once during application shutdown, after all
  macro-level operations have completed. It ensures a clean shutdown of the HexKL
  environment and prevents resource leaks.
 */
int hexkl_macro_finalize(void);

/*!
  @ingroup NPUMacroGeneral
  @brief Locks the HMX unit for exclusive use by the current process.

  This function acquires an exclusive lock on the HMX unit,
  preventing other processes from executing macro-level matrix operations concurrently.
  While locked, only the current process can invoke HexKL macro functions that utilize
  the HMX unit.

  @return
  - `AEE_SUCCESS` on successful lock acquisition.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADSTATE`, etc.) in case of failure.

  @note
  The HMX unit can be unlocked during runtime using `hexkl_macro_unlock_hmx()`.
  Once unlocked, macro-level operations may be shared across processes unless explicitly
  locked again.
 */
int hexkl_macro_lock_hmx(void);

/*!
  @ingroup NPUMacroGeneral
  @brief Unlocks the HMX unit, allowing shared access by other processes.

  This function releases the exclusive lock on the HMX unit
  previously acquired by the current process. Once unlocked, macro-level matrix operations
  via the HexKL API may be accessed by other processes.

  @return
  - `AEE_SUCCESS` on successful unlock.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADSTATE`, etc.) in case of failure.

  @note
  This function should be called when exclusive access to the HMX unit
  is no longer required. Re-locking is possible if exclusive macro functionality is needed again.
 */
int hexkl_macro_unlock_hmx(void);

/*!
  @ingroup NPUMacroGeneral
  @brief Retrieves the HexKL version string for the NPU environment.

  This function returns the current version of the HexKL library configured for NPU usage.
  The version string includes semantic versioning and platform details, formatted as:
  `MAJOR_MINOR_PATCH_<stage>_HEXAGON_<arch>`, e.g., `1_0_0_beta_HEXAGON_V73`.

  @param[out] version
  A pointer to a character buffer where the version string will be stored.
  The buffer must be large enough to hold the full version string.

  @return
  - `AEE_SUCCESS` on success.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`) in case of failure.

 */
int hexkl_macro_get_version(char* version);

/*!
  @ingroup HexKLNPUMacro
  @defgroup NPUMacroMatMul Matrix Multiplication Functions
  @brief Defines functions for performing matrix multiplication using the HMX unit.
*/
/*!
  @ingroup NPUMacroMatMul
  @brief
  Performs matrix multiplication of FP16 activations by FP16 weights, producing FP16 results. It avoids data layout and
  type conversion overhead, assuming the caller provides inputs in the expected formats:
  - `A`: output matrix, type FP16, layout AH (activation layout for HexKL)
  - `X`: input matrix, type FP16, layout AH (activation layout for HexKL)
  - `W`: weight matrix, type FP16, layout WH (weight layout for HexKL)

  @param[in]  n_row      Number of rows in matrix X and A.
  @param[in]  n_col      Number of columns in matrix W and A.
  @param[in]  n_inner    Shared dimension between X and W (columns of X, rows of W).
  @param[out] A          Pointer to the output matrix A (FP16, AH layout).
  @param[in]  X          Pointer to the input matrix X (FP16, AH layout).
  @param[in]  W          Pointer to the weight matrix W (FP16, WH layout).

  @return
  - `AEE_SUCCESS` on successful execution.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  This is the ideal kernel for FP16 matmul on Hexagon NPU, assuming the caller handles layout and type preparation.
 */
int hexkl_macro_mm_f16(
  int n_row,
  int n_col,
  int n_inner,
  _Float16* restrict A,
  const _Float16* restrict X,
  const _Float16* restrict W
);

/*!
  @ingroup NPUMacroMatMul
  @brief
  Performs matrix multiplication of ui8 activations by i8 weights, producing i32 results. It avoids data layout and type
  conversion overhead, assuming the caller provides inputs in the expected formats:
  - `A`: output matrix, type INT32
  - `X`: input matrix, type UINT8
  - `W`: weight matrix, type INT8

  @param[in]  n_row      Number of rows in matrix X and A.
  @param[in]  n_col      Number of columns in matrix W and A.
  @param[in]  n_inner    Shared dimension between X and W (columns of X, rows of W).
  @param[out] A          Pointer to the output matrix A (INT32).
  @param[in]  X          Pointer to the input matrix X (UINT8).
  @param[in]  W          Pointer to the weight matrix W (INT8).

  @return
  - `AEE_SUCCESS` on successful execution.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  This is the ideal kernel for A8W8I32 matmul on Hexagon NPU, assuming the caller handles layout and type preparation.
 */
int hexkl_macro_mm_u8i8_i32(
  int n_row,
  int n_col,
  int n_inner,
  int32_t* restrict A,
  const uint8_t* restrict X,
  const int8_t* restrict W
);

/*!
  @ingroup NPUMacroMatMul
  @brief
  Performs matrix multiplication of ui8 activations by i4 weights, producing i32 results. It avoids data layout and type
  conversion overhead, assuming the caller provides inputs in the expected formats:
  - `A`: output matrix, type INT32
  - `X`: input matrix, type UINT8 in row major format
  - `W`: weight matrix, type INT8 (packed i4 values) in WH format

  @param[in]  n_row      Number of rows in matrix X and A.
  @param[in]  n_col      Number of columns in matrix W and A.
  @param[in]  n_inner    Shared dimension between X and W (columns of X, rows of W).
  @param[out] A          Pointer to the output matrix A (INT32).
  @param[in]  X          Pointer to the input matrix X (UINT8).
  @param[in]  W          Pointer to the weight matrix W (INT8, containing packed i4 values).

  @return
  - `AEE_SUCCESS` on successful execution.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  This is the ideal kernel for A8W4I32 matmul on Hexagon NPU, assuming the caller handles layout and type preparation.
  The i4 weights are packed into i8 containers, with two i4 values per i8 byte.
 */
int hexkl_macro_mm_u8i4_i32(
  int n_row,
  int n_col,
  int n_inner,
  int32_t* restrict A,
  const uint8_t* restrict X,
  const int8_t* restrict W
);

/*!
  @ingroup HexKLNPUMacro
  @defgroup NPUMacroLayout Data Layout Transformation Functions
  @brief Defines functions for transforming between data layouts including row-major and HMX optimized formats.
*/
/*!
  @ingroup NPUMacroLayout
  @brief Recovers the row-major layout from AH (activation layout for the HMX unit) in-place on Hexagon.

  This function transforms the input matrix `A` from the AH layout
  back to standard row-major (RM) layout. The transformation is performed in-place,
  reversing the spatial reorganization applied for NPU execution.

  @param[in]      n_row Number of rows in the activation matrix.
                        Must be less than `HEXKL_N_ROW_BYTES_MAX`.
  @param[in]      n_col Number of columns in the activation matrix.
                        Must be less than `HEXKL_N_COL_BYTES_MAX`.
  @param[in,out]  A     Pointer to the activation matrix stored in AH layout.
                        The matrix is expected to be of size `n_row * n_col` and contain `_Float16` values.

  @return
  - `AEE_SUCCESS` on successful transformation.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.
*/
int hexkl_macro_ah_to_rm_f16_inplace(uint32_t n_row, uint32_t n_col, _Float16* restrict A);

/*!
  @ingroup NPUMacroLayout
  @brief Applies the AH (activation layout for the HMX unit) data layout to input activations in-place on Hexagon.

  This function transforms the input matrix `X` from row-major (RM) layout
  to the AH layout required by the HMX unit. The transformation is performed in-place,
  modifying the contents of `X` directly.

  The AH layout is optimized for NPU execution and must be applied before passing
  activations to HEXKL-based operators.

  @param[in]     n_row Number of rows in the input matrix.
                      Must be less than `HEXKL_N_ROW_BYTES_MAX`.
  @param[in]     n_col Number of columns in the input matrix.
                      Must be less than `HEXKL_N_INNER_BYTES_MAX`.
  @param[in,out] X     Pointer to the input matrix stored in row-major order.
                      The matrix is expected to be of size `n_row * n_col` and contain `_Float16` values.

  @return
  - `AEE_SUCCESS` on successful transformation.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.
*/
int hexkl_macro_rm_to_ah_f16_inplace(uint32_t n_row, uint32_t n_col, _Float16* restrict X);

/*!
  @ingroup NPUMacroLayout
  @brief Applies the WH (weight layout for the HMX unit) data layout to weights in-place on Hexagon.

  This function transforms the input weight matrix `W` from row-major (RM) layout
  to the WH layout required by the HMX unit. The transformation is performed
  in-place and prepares the weights for optimized execution on the NPU.

  @param[in]     n_row Number of rows in the input weight matrix.
                      Must be less than `HEXKL_N_COL_BYTES_MAX`.
  @param[in]     n_col Number of columns in the input weight matrix.
                      Must be less than `HEXKL_N_INNER_BYTES_MAX`.
  @param[in,out] W     Pointer to the input weight matrix stored in row-major order.
                      The matrix is expected to be of size `n_row * n_col` and contain `_Float16` values.

  @return
  - `AEE_SUCCESS` on successful transformation.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.
*/
int hexkl_macro_rm_to_wh_f16_inplace(size_t n_row, size_t n_col, _Float16* restrict W);

#endif // #ifdef __hexagon__

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //__HEXKL_MACRO_H__
