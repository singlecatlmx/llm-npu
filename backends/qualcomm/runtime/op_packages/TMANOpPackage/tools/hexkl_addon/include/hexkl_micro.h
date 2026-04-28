// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

#ifndef __HEXKL_MICRO_H__
#define __HEXKL_MICRO_H__

#include <stdint.h>

/*!
  @file hexkl_micro.h
  @brief Defines constants and functions in the HexKL NPU Micro API.
*/

/*!
  @defgroup HexKLNPUMicro HexKL NPU Micro API
  @brief Low-level NPU functions for matrix multiplication, accumulator access, and layout transformation.
*/

#ifdef __cplusplus
extern "C" {
#endif

/*!
  @ingroup HexKLNPUMicro
  @defgroup NPUMicroConstants Constants
  @brief Defines constants for tensor sizes, memory alignment, and version string lengths.
*/

/*!
  @ingroup NPUMicroConstants
  @def HEXKL_HMX_INT8_BLOCK_N_COL
  @brief Number of columns in an INT8 block for HMX processing.

  This macro defines the number of columns (N dimension) used in an INT8 block
  for HMX matrix operations.
*/
#define HEXKL_HMX_INT8_BLOCK_N_COL (32U)

/*!
  @ingroup NPUMicroConstants
  @def HEXKL_HMX_INT8_BLOCK_N_ROW
  @brief Number of rows in an INT8 block for HMX processing.

  This macro defines the number of rows (M dimension) used in an INT8 block
  for HMX matrix operations.
*/
#define HEXKL_HMX_INT8_BLOCK_N_ROW (64U)

/*!
  @ingroup NPUMicroConstants
  @def HEXKL_HMX_INT8_BLOCK_N_INNER
  @brief Inner dimension size for INT8 block multiplication in HMX.

  This macro defines the inner dimension (K) used in block-based matrix
  multiplication for INT8 data types in HMX.
*/
#define HEXKL_HMX_INT8_BLOCK_N_INNER (32U)

/*!
  @ingroup NPUMicroConstants
  @def HEXKL_HMX_F16_BLOCK_N_COL
  @brief Number of columns in an F16 block for HMX processing.

  This macro defines the number of columns (N dimension) used in an F16 block
  for HMX matrix operations.
*/
#define HEXKL_HMX_F16_BLOCK_N_COL (32U)

/*!
  @ingroup NPUMicroConstants
  @def HEXKL_HMX_F16_BLOCK_N_ROW
  @brief Number of rows in an F16 block for HMX processing.

  This macro defines the number of rows (M dimension) used in an F16 block
  for HMX matrix operations.
*/
#define HEXKL_HMX_F16_BLOCK_N_ROW (32U)

/*!
  @ingroup NPUMicroConstants
  @def HEXKL_HMX_F16_BLOCK_N_INNER
  @brief Inner dimension size for F16 block multiplication in HMX.

  This macro defines the inner dimension (K) used in block-based matrix
  multiplication for F16 data types in HMX.
*/
#define HEXKL_HMX_F16_BLOCK_N_INNER (32U)

/*!
  @ingroup NPUMicroConstants
  @def HEXKL_HMX_ACTIVATION_ALIGNMENT
  @brief Byte alignment requirement for activation matrices in VTCM.

  This macro specifies the required byte alignment (2048 bytes) for activation
  matrices when stored in VTCM. Proper alignment
  ensures optimal memory access patterns and performance during HMX operations.
*/
#define HEXKL_HMX_ACTIVATION_ALIGNMENT (2048U)

/*!
  @ingroup NPUMicroConstants
  @def HEXKL_HMX_WEIGHTS_ALIGNMENT
  @brief Byte alignment requirement for weights matrices in VTCM.

  This macro defines the required byte alignment (128 bytes) for weights
  matrices stored in VTCM. Ensuring proper
  alignment improves memory access efficiency and supports optimized
  computation in HMX-based operations.
*/
#define HEXKL_HMX_WEIGHTS_ALIGNMENT (128U)

/*!
  @ingroup NPUMicroConstants
  @def HEXKL_HMX_CONFIG_ALIGNMENT
  @brief Byte alignment requirement for HMX configuration in VTCM.

This macro defines the required byte alignment (256 bytes) for HMX configuration
stored in VTCM. Proper alignment ensures efficient
memory access and supports optimized performance in HMX computations.
*/
#define HEXKL_HMX_CONFIG_ALIGNMENT (256U)

/*!
  @ingroup NPUMicroConstants
  @def HEXKL_PREREL_STR_LEN
  @brief Length of buffer required to store the pre-release string.

  This macro defines the size of the character buffer (in bytes) needed to hold
  the pre-release identifier string (e.g., "alpha", "beta") returned by
  `hexkl_micro_get_version()`. The value is set to 32 bytes to ensure sufficient
  space for typical pre-release labels.
*/
#define HEXKL_PREREL_STR_LEN (32U)

/*!
  @ingroup HexKLNPUMicro
  @defgroup NPUMicroGeneral General Functions
  @brief Defines general functions
*/
/*!
  @ingroup NPUMicroGeneral
  @brief Retrieves the HexKL library version and its target Hexagon architecture.

  This function obtains the current version of the HexKL library configured for NPU usage,
  including its semantic version components (`major`, `minor`, `patch`) and an optional
  pre-release stage identifier (e.g., `"beta"`). It also retrieves the Hexagon architecture
  version for which the HexKL library was compiled

  The version string constructed from these components typically follows the format:
  `MAJOR_MINOR_PATCH_<stage>_HEXAGON_<arch>`, e.g., `1_0_0_beta_HEXAGON_V73`.

  @param[out] major
  Pointer to integer major version number.

  @param[out] minor
  Pointer to integer minor version number.

  @param[out] patch
  Pointer to integer patch version number.

  @param[out] version_prerel
  Pointer to a character buffer with the pre-release stage string
  (e.g., `"beta"`, `"alpha"`). The buffer must be at least `HEXKL_PREREL_STR_LEN`
  bytes long to accommodate the string, including the null terminator.

  @param[out] hex_version
  Pointer to integer Hexagon architecture version (e.g., `73` for V73).

  @return
  - `AEE_SUCCESS` on success.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`) in case of failure.
*/
int hexkl_micro_get_version(int* major, int* minor, int* patch, char* version_prerel, int* hex_version);

/*!
  @ingroup NPUMicroGeneral
  @brief
  Initializes and powers up hardware components including HVX, HMX, VTCM, and DCVS.

  This function serves as a reference initializer for micro API examples.
  NPU programmers using the `hexkl_micro.a` library may implement their own
  initialization routines tailored to their specific use cases.

  @param[out] vtcm_base
  Pointer to the start address of the VTCM memory region.

  @param[out] vtcm_size
  Pointer to a variable that will receive the size of the VTCM memory region.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.
*/
int hexkl_micro_hw_init(uint8_t** vtcm_base, uint32_t* vtcm_size);

/*!
  @ingroup NPUMicroGeneral
  @brief
  Locks the HMX unit for exclusive use by the current process.

  This function is primarily used for testing and example purposes within the
  `hexkl_micro.a` API. NPU programmers integrating this library may implement
  their own HMX locking routines suited to their application needs.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.
*/
int hexkl_micro_hmx_lock(void);

/*!
  @ingroup NPUMicroGeneral
  @brief
  Unlocks the HMX unit, releasing exclusive access held by the current process.

  This function is primarily used for testing and example purposes within the
  `hexkl_micro.a` API. NPU programmers integrating this library may implement
  their own HMX unlocking routines suited to their application needs.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.
*/
int hexkl_micro_hmx_unlock(void);

/*!
  @ingroup NPUMicroGeneral
  @brief Returns the size of VTCM memory required to store HMX configuration.

  This function returns the number of contiguous bytes in VTCM memory needed
  to store HMX configuration information.

  @return
  Size in bytes required for HMX register configuration.
*/
uint32_t hexkl_micro_hmx_config_size(void);

/*!
  @ingroup HexKLNPUMicro
  @defgroup NPUMicroAccumulator Accumulator Functions
  @brief Defines functions for configuring, accessing, and manipulating HMX accumulators.
*/
/*!
  @ingroup NPUMicroAccumulator
  @brief
  Sets up the HMX unit for reading from the int32 accumulator.

  This function initializes the configuration region in VTCM required for accessing
  the int32 accumulator. The configuration data is system-managed, but the user controls
  its location via `hmx_config_offset`.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] hmx_config_offset
  Byte offset from `vtcm_base` to the start of the HMX configuration region.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + hmx_config_offset` must be aligned to ::HEXKL_HMX_CONFIG_ALIGNMENT.
  - The region starting at `vtcm_base + hmx_config_offset` must span
    `hexkl_hmx_config_size()` bytes of contiguous reserved space.
  - Any non-system modification to the HMX config region invalidates the config
*/
int hexkl_micro_hmx_setup_acc_read_int32(uint8_t* vtcm_base, uint32_t hmx_config_offset);

/*!
  @ingroup NPUMicroAccumulator
  @brief
  Sets up the HMX unit for reading from the fp16 accumulator.

  This function initializes the configuration region in VTCM required for accessing
  the fp16 accumulator. The configuration data is system-managed, but the user controls
  its location via `hmx_config_offset`.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] hmx_config_offset
  Byte offset from `vtcm_base` to the start of the HMX configuration region.
  This region must span `hexkl_micro_hmx_config_size()` bytes.

  @details
  - This function must be called before invoking `hexkl_acc_read_f16()`.
  - It only needs to be called once, unless the configuration region is overwritten.
  - The configuration remains valid across multiple matrix multiply and accumulator operations.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + hmx_config_offset` must be aligned to ::HEXKL_HMX_CONFIG_ALIGNMENT.
  - The region starting at `vtcm_base + hmx_config_offset` must span
    `hexkl_hmx_config_size()` bytes of contiguous reserved space.
  - Any non-system modification to the HMX config region invalidates the config
*/
int hexkl_micro_hmx_setup_acc_read_f16(uint8_t* vtcm_base, uint32_t hmx_config_offset);

/*!
  @ingroup NPUMicroAccumulator
  @brief
  Clears the HMX fp16 accumulator.
*/
void hexkl_micro_hmx_acc_clear_f16(void);

/*!
  @ingroup NPUMicroAccumulator
  @brief
  Clears the HMX int32 accumulator.
*/
void hexkl_micro_hmx_acc_clear_int32(void);

/*!
  @ingroup NPUMicroAccumulator
  @brief
  Reads a 32×32 fp16 accumulator tile from the HMX unit and stores it in VTCM memory
  using fp16 activation layout.

  This function retrieves the contents of the fp16 accumulator and writes 2 KB of
  output starting at `vtcm_base[out_offset]`. The layout used is the activation layout
  expected by downstream operators.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] hmx_config_offset
  Byte offset from `vtcm_base` to the HMX configuration region. This must have been
  initialized by a prior call to `hexkl_micro_hmx_setup_acc_read_f16()`.

  @param[in] out_offset
  Byte offset from `vtcm_base` to the destination buffer for the accumulator data.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + out_offset` must be aligned to ::HEXKL_HMX_ACTIVATION_ALIGNMENT.
  - `hmx_config_offset` must point to a valid configuration region initialized by
    `hexkl_micro_hmx_setup_acc_read_f16()`.
 */
int hexkl_micro_hmx_acc_read_f16(uint8_t* vtcm_base, uint32_t hmx_config_offset, uint32_t out_offset);

/*!
  @ingroup NPUMicroAccumulator
  @brief
  Reads HMX accumulators and stores the result in VTCM memory.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] hmx_config_offset
  Byte offset from `vtcm_base` to the HMX configuration region in VTCM.

  @param[in] out_offset
  Byte offset from `vtcm_base` to the destination buffer for shuffled accumulators.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + hmx_config_offset` must be aligned to ::HEXKL_HMX_CONFIG_ALIGNMENT.
 */
int hexkl_micro_hmx_acc_read_int32(uint8_t* vtcm_base, uint32_t hmx_config_offset, uint32_t out_offset);

/*!
  @ingroup HexKLNPUMicro
  @defgroup NPUMicroMatMul Matrix Multiplication Functions
  @brief Defines functions for performing matrix multiplication using HMX.
*/
/*!
  @ingroup NPUMicroMatMul
  @brief
  Performs a 64×32 activation by 32×32 weight int8 matrix multiplication, adding the
  results into a 64×32 int32 accumulator.

  The left-hand side (LHS) activation tile must be located at `vtcm_base[activation_offset]`,
  representing a 64×32 matrix in flat row-major `uint8_t` layout.

  The right-hand side (RHS) weight tile must be located at `vtcm_base[weight_offset]`,
  representing a 32×32 matrix in `int8_t` layout.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory region.

  @param[in] activation_offset
  Byte offset from `vtcm_base` to the start of the activation matrix.

  @param[in] weight_offset
  Byte offset from `vtcm_base` to the start of the weight matrix.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + activation_offset` must be aligned to ::HEXKL_HMX_ACTIVATION_ALIGNMENT.
  - `vtcm_base + weight_offset` must be aligned to ::HEXKL_HMX_WEIGHTS_ALIGNMENT.
 */
int hexkl_micro_hmx_mm_u8i8(uint8_t* vtcm_base, uint32_t activation_offset, uint32_t weight_offset);

/*!
  @ingroup NPUMicroMatMul
  @brief
  Performs a 64x32 activation by 32x32 weight int4 matrix multiplication, adding the
  results into a 64x32 int32 accumulator.

  The left-hand side (LHS) activation tile must be located at `vtcm_base[activation_offset]`,
  representing a 64x32 matrix in flat row-major `uint8_t` layout.

  The right-hand side (RHS) weight tile must be located at `vtcm_base[weight_offset]`,
  representing a 32x32 matrix in packed `int4_t` layout (stored as `int8_t` with two 4-bit
  values per byte).

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory region.

  @param[in] activation_offset
  Byte offset from `vtcm_base` to the start of the activation matrix.

  @param[in] weight_offset
  Byte offset from `vtcm_base` to the start of the weight matrix.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + activation_offset` must be aligned to ::HEXKL_HMX_ACTIVATION_ALIGNMENT.
  - `vtcm_base + weight_offset` must be aligned to ::HEXKL_HMX_WEIGHTS_ALIGNMENT.
  - Weight tile size is 512 bytes (32x32 4-bit values packed into int8_t array).
  - Each byte in the weight matrix contains two 4-bit signed values.
 */
int hexkl_micro_hmx_mm_u8i4(uint8_t* vtcm_base, uint32_t activation_offset, uint32_t weight_offset);

/*!
  @ingroup NPUMicroMatMul
  @brief
  Performs a 32×32 by 32×32 fp16 matrix multiplication, adding the results into
  the 32×32 fp16 accumulator.

  The left-hand side (LHS) tile must be located at `vtcm_base[activation_offset]`,
  representing a 32×32 matrix in fp16 activation layout.

  The right-hand side (RHS) tile must be located at `vtcm_base[weight_offset]`,
  representing a 32×32 matrix in fp16 weight layout.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] activation_offset
  Byte offset from `vtcm_base` to the activation matrix tile.

  @param[in] weight_offset
  Byte offset from `vtcm_base` to the weight matrix tile.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + activation_offset` must be aligned to ::HEXKL_HMX_ACTIVATION_ALIGNMENT.
  - `vtcm_base + weight_offset` must be aligned to ::HEXKL_HMX_WEIGHTS_ALIGNMENT.
 */
int hexkl_micro_hmx_mm_f16(uint8_t* vtcm_base, uint32_t activation_offset, uint32_t weight_offset);

/*!
  @ingroup HexKLNPUMicro
  @defgroup NPUMicroLayout Data Layout Transformation Functions
  @brief Defines functions for transforming between data layouts including row-major and HMX optimized formats.
*/

/*!
  @ingroup NPUMicroLayout
  @brief
  Transforms a 32×32 fp16 tile from activation layout to flat row-major layout.

  This function is typically used to convert the layout of fp16 accumulator data
  into a flat format suitable for non-matrix-multiply operators or external processing.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] flat_out_offset
  Byte offset from `vtcm_base` to the destination buffer for the flat layout.

  @param[in] activation_in_offset
  Byte offset from `vtcm_base` to the source buffer containing the activation layout.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + flat_out_offset` must be aligned to 2048 bytes.
  - `vtcm_base + activation_in_offset` must be aligned to 2048 bytes.
 */
int hexkl_micro_hmx_ah_to_rm_f16(uint8_t* vtcm_base, uint32_t flat_out_offset, uint32_t activation_in_offset);

/*!
  @ingroup NPUMicroLayout
  @brief
  Transforms a 32×32 fp16 tile from flat row-major layout to activation layout.

  This function is typically used at runtime to prepare left-hand side (LHS) tiles
  for fp16 matrix multiplication. It converts a tile stored in flat layout into
  the activation layout expected by the HMX unit.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] activation_out_offset
  Byte offset from `vtcm_base` to the destination buffer for the activation layout.

  @param[in] flat_in_offset
  Byte offset from `vtcm_base` to the source buffer containing the flat layout.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + activation_out_offset` must be aligned to ::HEXKL_HMX_ACTIVATION_ALIGNMENT.
  - `vtcm_base + flat_in_offset` must be aligned to ::HEXKL_HMX_ACTIVATION_ALIGNMENT.
 */
int hexkl_micro_hmx_rm_to_ah_f16(uint8_t* vtcm_base, uint32_t activation_out_offset, uint32_t flat_in_offset);

/*!
  @ingroup NPUMicroLayout
  @brief
  Copies a 2048-element int8 weight tile into VTCM using the HMX-specific layout.

  This function transforms a flat int8 weight tile into the layout expected by the HMX unit
  for int8 matrix multiplication and writes it to VTCM memory at the specified offset.

  @details
  - `row_tile` and `col_tile` specify the tile indices to extract from `wt_old`.
  - `vtcm_base[weight_offset]` must point to the first element of a 2048-byte int8 tile.
  - `wt_cols` defines the number of columns in the original flat weight matrix.

  This function performs layout transformation "on the fly" and is intended to be used
  with int8-based HMX operations such as `hexkl_micro_hmx_matmul_i8()`.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] weight_offset
  Byte offset from `vtcm_base` to the destination buffer for the weight tile.

  @param[in] wt_old
  Pointer to the source int8 weight matrix in flat layout.

  @param[in] row_tile
  Row index of the tile to extract from `wt_old`.

  @param[in] col_tile
  Column index of the tile to extract from `wt_old`.

  @param[in] wt_cols
  Number of columns in the original flat weight matrix.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + weight_offset` must be aligned to ::HEXKL_HMX_WEIGHTS_ALIGNMENT.
 */
int hexkl_micro_hmx_rm_to_wh_i8(
  uint8_t* vtcm_base,
  uint32_t weight_offset,
  const int8_t* wt_old,
  uint32_t row_tile,
  uint32_t col_tile,
  uint32_t wt_cols
);

/*!
  @ingroup NPUMicroLayout
  @brief
  Copies a 512-byte int4 weight tile into VTCM using the HMX-specific layout.

  This function transforms a flat int4 weight tile into the layout expected by the HMX unit
  for int4 matrix multiplication and writes it to VTCM memory at the specified offset.

  @details
  - `row_tile` and `col_tile` specify the tile indices to extract from `wt_old`.
  - `vtcm_base[weight_offset]` must point to the first element of a 512-byte int4 tile.
  - `wt_cols` defines the number of columns in the original flat weight matrix.

  This function performs layout transformation "on the fly" and is intended to be used
  with int4-based HMX operations such as `hexkl_micro_hmx_mm_u8i4()`.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] weight_offset
  Byte offset from `vtcm_base` to the destination buffer for the weight tile.

  @param[in] wt_old
  Pointer to the source int4 weight matrix in flat layout (stored as int8_t with values in range [-8, 7]).

  @param[in] row_tile
  Row index of the tile to extract from `wt_old`.

  @param[in] col_tile
  Column index of the tile to extract from `wt_old`.

  @param[in] wt_cols
  Number of columns in the original flat weight matrix.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + weight_offset` must be aligned to ::HEXKL_HMX_WEIGHTS_ALIGNMENT.
  - Weight tile size is 512 bytes (32x32 4-bit values packed into uint8_t array).
 */
int hexkl_micro_hmx_rm_to_wh_i4(
  uint8_t* vtcm_base,
  uint32_t weight_offset,
  const int8_t* wt_old,
  uint32_t row_tile,
  uint32_t col_tile,
  uint32_t wt_cols
);

/*!
  @ingroup NPUMicroLayout
  @brief
  Copies a 2048-element FP16 weight tile into VTCM using the HMX-specific layout.

  This function transforms a flat FP16 weight tile into the layout expected by the HMX unit
  for FP16 matrix multiplication and writes it to VTCM memory at the specified offset.

  @details
  - `row_tile` and `col_tile` specify the tile indices to extract from `wt_old`.
  - `vtcm_base[weight_offset]` must point to the first element of a 2048-byte FP16 tile.
  - `wt_cols` defines the number of columns in the original flat weight matrix.

  This function performs layout transformation "on the fly" and is intended to be used
  with FP16-based HMX operations such as `hexkl_micro_hmx_matmul_fp16()`.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] weight_offset
  Byte offset from `vtcm_base` to the destination buffer for the weight tile.

  @param[in] wt_old
  Pointer to the source FP16 weight matrix in flat layout.

  @param[in] row_tile
  Row index of the tile to extract from `wt_old`.

  @param[in] col_tile
  Column index of the tile to extract from `wt_old`.

  @param[in] wt_cols
  Number of columns in the original flat weight matrix.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + weight_offset` must be aligned to ::HEXKL_HMX_WEIGHTS_ALIGNMENT.
 */
int hexkl_micro_hmx_rm_to_wh_f16(
    uint8_t* __restrict vtcm_base,
    uint32_t weight_offset,
    const _Float16* __restrict wt_old,
    uint32_t row_tile,
    uint32_t col_tile,
    uint32_t wt_cols
);

/*!
  @ingroup HexKLNPUMicro
  @defgroup NPUMicroDataMove Data Movement Functions
  @brief Defines functions for copying data between DDR and VTCM.
*/
/*!
  @ingroup NPUMicroDataMove
  @brief
  Copies a preprocessed 32×32 int8 weight tile into VTCM memory using the correct
  int8 weight layout, from any external memory (e.g., DDR).

  This function is primarily intended for testing and debugging purposes. In operational
  mode, DMA is typically used for transferring weights.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] out_offset
  Byte offset from `vtcm_base` to the destination location for the weight tile.
  `vtcm_base[out_offset]` must point to the first element of a 32×32 int8 tile
  where the extracted tile will be stored.

  @param[in] input_matrix
  Pointer to the source matrix containing activation data, located in any memory
  (e.g., DDR). The matrix must be zero-padded to full tile size on both dimensions.

  @param[in] tile_row
  Row index of the tile to extract from `input_matrix`.

  @param[in] tile_col
  Column index of the tile to extract from `input_matrix`.

  @param[in] input_rows
  Number of valid (non-padded) rows in `input_matrix`.

  @param[in] input_cols
  Number of valid (non-padded) columns in `input_matrix`.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + out_offset` must be aligned to ::HEXKL_HMX_WEIGHTS_ALIGNMENT.
  - `input_matrix` must be zero-padded to full tile size on both dimensions.
  - Padding elements are not included in `input_rows` and `input_cols`.
  - `tile_row * HEXKL_HMX_INT8_BLOCK_N_COL < input_rows`
  - `tile_col * HEXKL_HMX_INT8_BLOCK_N_COL < input_cols`
 */
int hexkl_micro_hmx_copy_psubmatrix_to_8b_weight(
  uint8_t* vtcm_base,
  uint32_t out_offset,
  int8_t* input_matrix,
  uint32_t tile_row,
  uint32_t tile_col,
  uint32_t input_rows,
  uint32_t input_cols
);

/*!
  @ingroup NPUMicroDataMove
  @brief
  Copies a preprocessed 32×32 uint8 activation tile into VTCM memory using the correct
  uint8 activation layout, from any external memory (e.g., DDR).

  This function is primarily intended for testing and debugging purposes. In operational
  mode, DMA is typically used for transferring activations.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] out_offset
  Byte offset from `vtcm_base` to the destination location for the activation tile.
  `vtcm_base[out_offset]` must point to the first element of a 32×32 uint8 tile
  where the extracted tile will be stored.

  @param[in] input_matrix
  Pointer to the source matrix containing activation data, located in any memory
  (e.g., DDR). The matrix must be zero-padded to full tile size on both dimensions.

  @param[in] tile_row
  Row index of the tile to extract from `input_matrix`.

  @param[in] tile_col
  Column index of the tile to extract from `input_matrix`.

  @param[in] input_rows
  Number of valid (non-padded) rows in `input_matrix`.

  @param[in] input_cols
  Number of valid (non-padded) columns in `input_matrix`.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + out_offset` must be aligned to ::HEXKL_HMX_ACTIVATION_ALIGNMENT.
  - `input_matrix` must be zero-padded to full tile size on both dimensions.
  - Padding elements are not included in `input_rows` and `input_cols`.
  - `tile_row * HEXKL_HMX_INT8_BLOCK_N_ROW < input_rows`
  - `tile_col * HEXKL_HMX_INT8_BLOCK_N_COL < input_cols`
 */
int hexkl_micro_hmx_copy_submatrix_to_8b_activation(
  uint8_t* vtcm_base,
  uint32_t out_offset,
  const uint8_t* input_matrix,
  uint32_t tile_row,
  uint32_t tile_col,
  uint32_t input_rows,
  uint32_t input_cols
);

/*!
  @ingroup NPUMicroDataMove
  @brief
  Copies the shuffled result of HMX matrix multiplication from VTCM to DDR.

  This function transfers a 32-bit accumulator tile from VTCM memory to a submatrix
  in DDR memory. The tile is placed at the specified location in the output matrix,
  based on the provided tile row and column indices.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] in_offset
  Byte offset from `vtcm_base` to the start of the 32-bit accumulator tile.

  @param[out] output_matrix
  Pointer to the destination matrix in DDR memory where the tile will be copied.

  @param[in] tile_row
  Row index in `output_matrix` where the tile will be placed.

  @param[in] tile_col
  Column index in `output_matrix` where the tile will be placed.

  @param[in] output_rows
  Number of valid rows in `output_matrix`.

  @param[in] output_cols
  Number of valid columns in `output_matrix`.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.
 */
int hexkl_micro_hmx_copy_32b_to_submatrix(
  uint8_t* vtcm_base,
  uint32_t in_offset,
  int32_t* output_matrix,
  uint32_t tile_row,
  uint32_t tile_col,
  uint32_t output_rows,
  uint32_t output_cols
);

/*!
  @ingroup NPUMicroDataMove
  @brief
  Copies a preprocessed 32×32 fp16 weight tile into VTCM using the correct fp16 weight layout.

  This function extracts a tile from the input matrix located in DDR and writes it to
  VTCM memory at the specified offset, using the layout expected by the HMX unit for
  fp16 matrix multiplication.

  @details
  - `tile_row` and `tile_col` specify the tile indices to extract from `input_matrix`.
  - `vtcm_base[out_offset]` must point to the first element of a 32×32 fp16 tile.
  - `input_rows` and `input_cols` define the valid dimensions of `input_matrix`.

  This function should be used in conjunction with `hexkl_micro_hmx_layout_fp16_flat_to_weight()`,
  which preprocesses the weight matrix into the correct layout required by this function
  and by `hexkl_micro_hmx_matmul_fp16()`.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] out_offset
  Byte offset from `vtcm_base` to the destination buffer for the weight tile.

  @param[in] input_matrix
  Pointer to the source fp16 matrix in DDR memory.

  @param[in] tile_row
  Row index of the tile to extract from `input_matrix`.

  @param[in] tile_col
  Column index of the tile to extract from `input_matrix`.

  @param[in] input_rows
  Number of valid rows in `input_matrix`.

  @param[in] input_cols
  Number of valid columns in `input_matrix`.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + out_offset` must be aligned to ::HEXKL_HMX_WEIGHTS_ALIGNMENT.
  - `input_matrix` must be zero-padded to full tile size on both dimensions.
  - Padding elements are not included in `input_rows` and `input_cols`.
  - `tile_row * 32 < input_rows`
  - `tile_col * 32 < input_cols`
 */
int hexkl_micro_hmx_copy_psubmatrix_to_f16_weight(
  uint8_t* vtcm_base,
  uint32_t out_offset,
  const _Float16* input_matrix,
  uint32_t tile_row,
  uint32_t tile_col,
  uint32_t input_rows,
  uint32_t input_cols
);

/*!
  @ingroup NPUMicroDataMove
  @brief
  Copies a 32×32 fp16 submatrix from a full flat row-major matrix in DDR to VTCM.

  This function extracts a 32×32 tile from the input matrix in DDR and stores it
  in VTCM memory at the specified offset using flat row-major layout. It is typically
  used at the start of or during matrix multiplication operations to move activation
  tiles into VTCM.

  @details
  - `tile_row` and `tile_col` specify the tile indices to extract from `input_matrix`.
  - `vtcm_base[out_offset]` must point to the first element of a 32×32 fp16 tile.
  - `input_rows` and `input_cols` define the valid dimensions of `input_matrix`.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] out_offset
  Byte offset from `vtcm_base` to the destination buffer for the tile.

  @param[in] input_matrix
  Pointer to the source fp16 matrix in DDR memory.

  @param[in] tile_row
  Row index of the tile to extract from `input_matrix`.

  @param[in] tile_col
  Column index of the tile to extract from `input_matrix`.

  @param[in] input_rows
  Number of valid rows in `input_matrix`.

  @param[in] input_cols
  Number of valid columns in `input_matrix`.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + out_offset` must be aligned to ::HEXKL_HMX_ACTIVATION_ALIGNMENT.
  - `tile_row * 32 < input_rows`
  - `tile_col * 32 < input_cols`
 */
int hexkl_micro_hmx_copy_submatrix_to_f16(
  uint8_t* vtcm_base,
  uint32_t out_offset,
  const _Float16* input_matrix,
  uint32_t tile_row,
  uint32_t tile_col,
  uint32_t input_rows,
  uint32_t input_cols
);

/*!
  @ingroup NPUMicroDataMove
  @brief
  Copies a 32×32 fp16 tile in flat row-major layout from VTCM to DDR memory.

  This function transfers a 32×32 fp16 tile from VTCM (starting at `in_offset`)
  to the specified location in the output matrix in DDR. The destination tile
  is placed at the position defined by `tile_row` and `tile_col` in the
  `output_matrix`.

  @details
  - `tile_row` and `tile_col` specify the tile indices in `output_matrix`.
  - `vtcm_base[in_offset]` must point to the first element of a 32×32 fp16 tile.
  - `output_rows` and `output_cols` define the dimensions of the output matrix.

  This function is typically called after reading fp16 accumulator values into
  VTCM, to write the results back to DDR.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] in_offset
  Byte offset from `vtcm_base` to the source fp16 tile in flat layout.

  @param[out] output_matrix
  Pointer to the destination matrix in DDR memory.

  @param[in] tile_row
  Row index of the tile in `output_matrix`.

  @param[in] tile_col
  Column index of the tile in `output_matrix`.

  @param[in] output_rows
  Number of valid rows in `output_matrix`.

  @param[in] output_cols
  Number of valid columns in `output_matrix`.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + in_offset` must be aligned to ::HEXKL_HMX_ACTIVATION_ALIGNMENT.
  - `tile_row * 32 < output_rows`
  - `tile_col * 32 < output_cols`
 */
int hexkl_micro_hmx_copy_f16_to_submatrix(
  uint8_t* vtcm_base,
  uint32_t in_offset,
  _Float16* output_matrix,
  uint32_t tile_row,
  uint32_t tile_col,
  uint32_t output_rows,
  uint32_t output_cols
);

/*!
  @ingroup NPUMicroDataMove
  @brief
  Copies a 32×32 fp16 tile in flat row-major layout from VTCM to DDR as a 32×32 fp32 tile.

  This function reads a 32×32 fp16 tile from VTCM memory (starting at `in_offset`),
  converts each element to fp32, and writes the result to the specified location
  in the output matrix in DDR. The destination tile is placed at the position
  defined by `tile_row` and `tile_col` in `output_matrix`.

  @details
  - `tile_row` and `tile_col` specify the tile indices in `output_matrix`.
  - `vtcm_base[in_offset]` must point to the first element of a 32×32 fp16 tile.
  - `output_rows` and `output_cols` define the dimensions of the output matrix.

  This function is typically called after reading fp16 accumulator values into
  VTCM, to write the results back to DDR in fp32 format.

  @param[in] vtcm_base
  Pointer to the base of the allocated VTCM memory.

  @param[in] in_offset
  Byte offset from `vtcm_base` to the source fp16 tile in flat layout.

  @param[out] output_matrix
  Pointer to the destination matrix in DDR memory (fp32 format).

  @param[in] tile_row
  Row index of the tile in `output_matrix`.

  @param[in] tile_col
  Column index of the tile in `output_matrix`.

  @param[in] output_rows
  Number of valid rows in `output_matrix`.

  @param[in] output_cols
  Number of valid columns in `output_matrix`.

  @return
  Status code defined in "AEEStdErr.h":
  - `AEE_SUCCESS` on success.
  - Other error codes (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) on failure.

  @note
  - `vtcm_base + in_offset` must be aligned to ::HEXKL_HMX_ACTIVATION_ALIGNMENT.
  - `tile_row * 32 < output_rows`
  - `tile_col * 32 < output_cols`
 */
int hexkl_micro_hmx_copy_f16_to_f32_submatrix(
  uint8_t* vtcm_base,
  uint32_t in_offset,
  float* output_matrix,
  uint32_t tile_row,
  uint32_t tile_col,
  uint32_t output_rows,
  uint32_t output_cols
);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //__HEXKL_MICRO_H__
