// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

#ifndef __SDKL_H__
#define __SDKL_H__

#include <stdint.h>

/*!
  @file sdkl.h
  @brief Defines constants, types, enumerations, and functions in the HexKL CPU Macro API.
*/

/*!
  @defgroup HexKLCPUMacro HexKL CPU Macro API
  @brief Constants, types, and functions used for layout transformation and HMX-accelerated matrix multiplication.
*/

#ifdef __cplusplus
extern "C" {
#endif

/*!
  @ingroup HexKLCPUMacro
  @defgroup CPUMacroConstants Constants
  @brief Defines constants for version string buffer size and maximum tensor dimensionality.
*/

/*!
  @ingroup CPUMacroConstants
  @def SDKL_VERSION_STR_LEN
  @brief Minimum buffer size required to store the SDKL version string.

  This macro defines the minimum length (in bytes) of the character buffer
  that must be provided to the function `sdkl_npu_get_version(char *version)`.
  The buffer should be at least this size to ensure the version string is
  safely written without overflow.

  @note The version string is null-terminated.
 */
#define SDKL_VERSION_STR_LEN (128U)

/*!
  @ingroup CPUMacroConstants
  @def SDKL_TENSOR_MAX_NUM_OF_DIMS
  @brief Maximum number of dimensions supported by SDKL tensors.

  This macro defines the upper limit on the number of dimensions a tensor
  can have in SDKL. It is used to allocate fixed-size arrays for shape and
  stride metadata, and to validate tensor dimensionality during creation
  and processing.

  @note Tensors exceeding this dimensionality are not supported and may
  result in an error or undefined behavior.
*/
#define SDKL_TENSOR_MAX_NUM_OF_DIMS (2U)

/*!
  @ingroup HexKLCPUMacro
  @defgroup CPUMacroTypes Types
  @brief Defines execution platforms, tensor data types, quantization formats, memory layouts, and tensor metadata
  structures.
*/

/*!
  @ingroup CPUMacroTypes
  @enum sdkl_tensor_platform_e
  @brief Enumeration of supported execution platforms for tensor operations in SDKL.

  This enum defines the available hardware platforms on which tensor operations can be executed.
  Selection of platform may depend on performance, power efficiency, or hardware availability.

  @note The number and identity of NPUs (e.g., NPU0, NPU1) are **platform-dependent** and may
  vary across Snapdragon devices. It is up to the programmer to know how many NPUs are available.

  @note The beta version of SDKL supports only NPU0. Future releases may add support for additional platforms
*/
typedef enum {
  /*!
   @brief Execute tensor operations on the CPU.
  */
  SDKL_PLATFORM_CPU = 0,

  /*!
   @brief Execute tensor operations on the GPU (Graphics Processing Unit).
  */
  SDKL_PLATFORM_GPU = 1,
  /*!
   @brief Execute tensor operations on the NPU0 (Neural Processing Unit #0).
  */
  SDKL_PLATFORM_NPU0 = CDSP_DOMAIN_ID,

  /*!
   @brief Execute tensor operations on the NPU1 (Neural Processing Unit #1).
  */
  SDKL_PLATFORM_NPU1 = CDSP1_DOMAIN_ID,

  /*!
   @brief Invalid platform selection. Also used to indicate the number of valid values.
  */
  SDKL_PLATFORM_INVALID
} sdkl_tensor_platform_e;

/*!
  @ingroup CPUMacroTypes
  @enum sdkl_tensor_quantization_e
  @brief Enumeration of supported tensor quantization formats in SDKL.

  This enum defines supported tensor data quantization formats.
*/
typedef enum {
  /*!
   @brief The tensor is not quantized.
  */
  SDKL_QUANT_NONE,

  /*!
   @brief Invalid quantization.
  */
  SDKL_QUANT_INVALID
} sdkl_tensor_quantization_e;

/*!
  @ingroup CPUMacroTypes
  @enum sdkl_tensor_dtype_e
  @brief Enumeration of supported tensor data types in SDKL.

  This enum defines the scalar types that tensors can use for storage and computation.
  These types include integer, unsigned integer, and floating-point formats with varying precision.
*/
typedef enum {
  /*!
    @brief 4-bit signed integer.
  */
  SDKL_DTYPE_I4,

  /*!
    @brief 8-bit signed integer.
  */
  SDKL_DTYPE_I8,

  /*!
   @brief 8-bit unsigned integer.
  */
  SDKL_DTYPE_U8,

  /*!
   @brief 32-bit signed integer.
  */
  SDKL_DTYPE_I32,

  /*!
   @brief 16-bit floating-point (IEEE 754 half precision).
  */
  SDKL_DTYPE_FP16,

  /*!
   @brief 32-bit floating-point (IEEE 754 single precision).
  */
  SDKL_DTYPE_FP32,

  /*!
   @brief Invalid tensor type.
  */
  SDKL_DTYPE_INVALID
} sdkl_tensor_dtype_e;

/*!
  @ingroup CPUMacroTypes
  @enum sdkl_tensor_layout_e
  @brief Enumeration of supported tensor memory layouts in SDKL.

  This enum defines the memory layout formats for tensors in SDKL. It includes options for 1D and 2D
  tensors, with variants for standard and HMX-specific layouts.
*/
typedef enum {
  /*!
    @brief 1D tensor with linear memory layout.
  */
  SDKL_LAYOUT_1D_LINEAR,

  /*!
    @brief 2D tensor with row-major layout (C-style).
  */
  SDKL_LAYOUT_2D_ROW_MAJOR,

  /*!
    @brief 2D tensor with transposed layout (column-major).
  */
  SDKL_LAYOUT_2D_COL_MAJOR,

  /*!
    @brief 2D tensor with row-major layout for weights matrix optimized for the HMX unit.
  */
  SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX,

  /*!
    @brief 2D tensor with row-major layout for activation matrix optimized for the HMX unit.
  */
  SDKL_LAYOUT_2D_ROW_MAJOR_ACTIVATION_HMX,

  /*!
    @brief 2D tensor with transposed layout for weights matrix optimized for the HMX unit.
  */
  SDKL_LAYOUT_2D_COL_MAJOR_WEIGHTS_HMX,

  /*!
    @brief 2D tensor with transposed layout for activation matrix optimized for the HMX unit.
  */
  SDKL_LAYOUT_2D_COL_MAJOR_ACTIVATION_HMX,

  /*!
    @brief Invalid layout type. Also used to indicate the number of valid enum values.
  */
  SDKL_LAYOUT_INVALID
} sdkl_tensor_layout_e;

/*!
  @ingroup CPUMacroTypes
  @struct sdkl_tensor_t
  @brief Represents a tensor object in SDKL.

  This structure defines the metadata and memory layout for a tensor used in SDKL.
  It supports multiple data types, quantization formats, and layout configurations.
 */
typedef struct {
  /*!
    @brief Dimensions of the tensor.

    An array of size `SDKL_TENSOR_MAX_NUM_OF_DIMS` specifying the size of each dimension.
    For example, a 2D tensor with shape `[N_ROW, N_COL]` will have `dims[0] = N_ROW` and `dims[1] = N_COL`.
  */
  uint64_t dims[SDKL_TENSOR_MAX_NUM_OF_DIMS];

  /*!
    @brief Strides of the tensor.

    An array of size `SDKL_TENSOR_MAX_NUM_OF_DIMS` specifying the number of elements to skip
    in memory to move to the next element along each dimension.

    The stride for a dimension defines how far (in elements) you need to move in memory to
    access the next element along that axis.

    For a contiguous 1D tensor:
    - `strides[0] = 1` (each element is adjacent in memory)

    For a contiguous 2D tensor stored in row-major order (e.g., C arrays):
    - `strides[0] = N_COL` (to move to the next row)
    - `strides[1] = 1` (to move to the next column)

    For a contiguous 2D transposed tensor:
    - `strides[0] = 1`
    - `strides[1] = N_ROW`

    @note Strides are expressed in units of elements, not bytes.
  */
  uint64_t strides[SDKL_TENSOR_MAX_NUM_OF_DIMS];

  /*!
    @brief Total number of elements accessed by the tensor.

    This field represents the number of scalar elements that will be accessed
    during computation, taking into account the tensor's shape (`dims[]`),
    strides (`strides[]`), and the starting offset (`data_offset`).

    For contiguous tensors:
      num_elements = dims[0] * dims[1] * ... * dims[ndims - 1]

    For strided tensors:
      num_elements = 1 + Σ (strides[i] * (dims[i] - 1))

    @note This value should be updated whenever `dims[]`, `strides[]`, `ndims`,
          or `data_offset` changes.
  */
  uint64_t num_elements;

  /*!
    @brief Offset to the actual tensor data within the data buffer.

    Specifies the offset (in elements) from the start of the `data` pointer to the beginning
    of the tensor's data.
  */
  uint64_t data_offset;

  /*!
    @brief Pointer to the tensor data buffer.

    This points to the memory containing the tensor's actual data values.
  */
  void* data;

  /*!
    @brief Data type of the tensor elements.

    Specifies the scalar type used for storage and computation, such as SDKL_DTYPE_FP16 or SDKL_DTYPE_INT8.
  */
  sdkl_tensor_dtype_e data_dtype;

  /*!
    @brief Quantization format of the tensor.

    Indicates whether the tensor is quantized and the format used.
  */
  sdkl_tensor_quantization_e quantization;

  /*!
    @brief Memory layout of the tensor.

    Specifies how the tensor elements are arranged in memory. This affects access patterns,
    performance, and compatibility with hardware accelerators. The layout can be linear for 1D tensors,
    or row/column-major for 2D tensors. HMX-specific layouts are optimized for the HMX unit,
    which expects tensors in a particular memory format.
  */
  sdkl_tensor_layout_e layout;

  /*!
    @brief Number of dimensions in the tensor.

    Specifies how many dimensions the tensor has. This value must be greater than zero
    and less than or equal to `SDKL_TENSOR_MAX_NUM_OF_DIMS`.

    For example, a 2D tensor has `ndims = 2`, with corresponding entries in `dims[]` and `strides[]`.

    @note A value of zero is invalid and indicates an uninitialized or malformed tensor.
   */
  uint8_t ndims;

  /*!
    @brief Indicates whether the tensor is stored in a contiguous block of memory.

    A value of 1 means the tensor's data layout is contiguous in memory, starting from `data + data_offset`,
    and matches the expected layout for its shape and strides.

    A value of 0 means the tensor is strided, transposed, or otherwise non-contiguous.

    This flag is used by SDKL kernels to optimize memory access patterns.

    @note This field should be validated against `dims[]`, `strides[]`.
  */
  uint8_t is_continuous;

} sdkl_tensor_t;

/*!
  @ingroup CPUMacroTypes
  @brief Configuration structure for NPU initialization.

  This structure is reserved for future use and may contain
  configuration parameters required to initialize the NPU.
*/
typedef struct {

} sdkl_npu_init_config_t;

/*!
  @ingroup CPUMacroTypes
  @brief Information structure for NPU initialization.

  This structure is reserved for future use and may contain
  runtime or hardware-specific information obtained during NPU initialization.
*/
typedef struct {

} sdkl_npu_init_info_t;

/*!
  @ingroup HexKLCPUMacro
  @defgroup CPUMacroGeneral General Functions
  @brief Defines general functions.
*/
/*!
  @ingroup CPUMacroGeneral
  @brief Initializes the SDKL library for NPU usage.

  This function sets up internal resources and prepares the SDKL library
  for subsequent operations involving the NPU. It must be called before
  any other SDKL-related functions.

  By default, this function locks the HMX unit for exclusive use
  by the current process. This ensures that matrix multiplication operations
  are isolated and not interfered with by other processes.

  @param[in] domain
  Specifies the target NPU domain for initialization. Valid values include:
  - `CDSP_DOMAIN_ID` for NPU0
  This constant is defined in `remote.h` from the Hexagon SDK and determines
  which compute DSP (CDSP) core the SDKL library will interact with.

  @param[in] config
  Pointer to a configuration structure for NPU initialization.
  May be `NULL`, in which case default initialization parameters will be used.

  @param[out] info
  Pointer to a structure that will be populated with initialization information.
  May be `NULL`, in which case no information will be returned.

  @return
  - `AEE_SUCCESS` on successful initialization.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_ENOMEM`, etc.) in case of failure.

  @note
  This function should be called once during application startup.
  The HMX unit remains locked until explicitly unlocked using
  `sdkl_npu_unlock_hmx()` or implicitly unlocked by calling `sdkl_npu_finalize(domain)`.
  Note that `sdkl_npu_finalize(domain)` also deinitializes other internal resources,
  and should only be used when the SDKL library is no longer needed.
 */
int sdkl_npu_initialize(int domain, const sdkl_npu_init_config_t* config, sdkl_npu_init_info_t* info);

/*!
  @ingroup CPUMacroGeneral
  @brief Deinitializes the SDKL library and releases NPU-related resources.

  This function cleans up internal resources allocated during initialization
  and usage of the SDKL library for NPU operations. It should be called when
  NPU functionality is no longer required, typically during application shutdown.

  In addition to general resource cleanup, this function also unlocks the
  HMX unit if it was previously locked by `sdkl_npu_initialize()` or
  `sdkl_npu_lock_hmx()`.

  @param[in] domain
  Specifies the target NPU domain for finalization. Valid values include:
  - `CDSP_DOMAIN_ID` for NPU0
  This constant is defined in `remote.h` from the Hexagon SDK and determine
  which compute DSP (CDSP) core the SDKL library will interact with.

  @return
  - `AEE_SUCCESS` on successful deinitialization.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADSTATE`, etc.) in case of failure.

  @note
  This function should be called once during application shutdown, after all
  NPU-related operations have completed. Unlocking the HMX unit via
  this function also results in finalization of other SDKL resources, so it
  should only be used when the SDKL library is no longer needed.
 */
int sdkl_npu_finalize(int domain);

/*!
  @ingroup CPUMacroGeneral
  @brief Locks the HMX unit for exclusive use by the current process.

  This function prevents other processes from accessing the HMX unit
  by acquiring an exclusive lock. While the device is locked, only the current
  process can perform matrix multiplication operations using the SDKL API.

  @param[in] domain
  Specifies the target NPU domain for locking. Valid values include:
  - `CDSP_DOMAIN_ID` for NPU0
  This constant is defined in `remote.h` from the Hexagon SDK and determine
  which compute DSP (CDSP) core the SDKL library will interact with.

  @return
  - `AEE_SUCCESS` on successful lock acquisition.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADSTATE`, etc.) in case of failure.

  @note
  The device can be unlocked during runtime. Once unlocked, only non-HMX
  functions from the SDKL API may be used until the device is locked again.
 */
int sdkl_npu_lock_hmx(int domain);

/*!
  @ingroup CPUMacroGeneral
  @brief Unlocks the HMX unit, allowing shared access by other processes.

  This function releases the exclusive lock on the HMX unit previously
  acquired by the current process. Once unlocked, matrix multiplication operations
  via the SDKL API are no longer permitted, but other non-HMX functions
  remain available.

  @param[in] domain
  Specifies the target NPU domain for unlocking. Valid values include:
  - `CDSP_DOMAIN_ID` for NPU0
  This constant is defined in `remote.h` from the Hexagon SDK and determine
  which compute DSP (CDSP) core the SDKL library will interact with.

  @return
  - `AEE_SUCCESS` on successful unlock.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADSTATE`, etc.) in case of failure.

  @note
  This function should be called when exclusive access to the HMX unit
  is no longer required. Re-locking is possible if HMX functionality is
  needed again.
 */

int sdkl_npu_unlock_hmx(int domain);

/*!
  @ingroup CPUMacroGeneral
  @brief Allocates a shared CPU-NPU buffer.

  This function allocates a memory buffer that is accessible by both the CPU and the NPU.
  The allocated buffer address is returned via the `buffer_ptr` parameter.

  @param[in]  size        Size of the buffer to allocate, in bytes.
  @param[out] buffer_ptr  Pointer to a location where the allocated buffer address will be stored.

  @return
  - `AEE_SUCCESS` on successful allocation.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_ENOMEM`, etc.) in case of failure.

  @note
  The caller is responsible for freeing the allocated buffer using the appropriate SDKL deallocation function.
 */
int sdkl_npu_alloc(size_t size, void** buffer_ptr);

/*!
  @ingroup CPUMacroGeneral
  @brief Frees a previously allocated shared CPU-NPU buffer.

  This function releases memory that was allocated using `sdkl_npu_alloc`.
  The buffer must be valid and not already freed. After this call, the pointer
  should not be used unless reallocated.

  @param[in] addr  Pointer to the buffer to be freed.

  @return
  - `AEE_SUCCESS` on successful deallocation.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EBADPARM`, `AEE_EFAILED`, etc.) in case of failure.

  @note
  The buffer must have been allocated using SDKL's allocation API.
 */
int sdkl_npu_free(void* addr);

/*!
  @ingroup CPUMacroGeneral
  @brief Retrieves the SDKL version string for the NPU environment.

  This function returns the current version of the SDKL library configured for NPU usage.
  The version string includes semantic versioning and platform details, formatted as:
  `MAJOR_MINOR_PATCH_<stage>_HEXAGON_<arch>`, e.g., `1_0_0_beta_HEXAGON_V73`.

  Additionally, this function retrieves the Hexagon architecture of the current hardware
  and compares it to the architecture for which the loaded `hexkl_skel.so` library was compiled.
  If a mismatch is detected, a warning message is printed to alert the user.

  @param[in]  domain     Which compute DSP (CDSP) core the SDKL library will interact with.
  @param[out] version
  A pointer to a character buffer where the version string will be stored.
  The buffer must be at least ::SDKL_VERSION_STR_LEN bytes long to ensure safe storage
  of the full version string including the null terminator.

  @return
  - `AEE_SUCCESS` on success.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`) in case of failure.

  @note
  This function must be called **after** `sdkl_npu_initialize()` has successfully initialized the SDKL library.
  Calling it before initialization may result in undefined behavior or error codes.
 */
int sdkl_npu_get_version(int domain, char* version);

/*!
  @ingroup HexKLCPUMacro
  @defgroup CPUMacroMatMul Matrix Multiplication Functions
  @brief Defines functions for performing matrix multiplication using HMX.
*/

/*!
  @ingroup CPUMacroMatMul
  @brief
  Performs matrix multiplication using SDKL tensor descriptors.

  This function multiplies two tensors representing matrices and stores the result in a third tensor.
  It supports quantized and transposed layouts, and leverages SDKL's hardware-accelerated kernels
  when the tensor metadata matches expected formats.

  The function assumes:
  - `result_tensor`: output tensor, shape `[N_ROW, N_COL]`, layout and type defined by `sdkl_tensor_t`
  - `left_tensor`: input tensor, shape `[N_ROW, N_INNER]`, layout and type defined by `sdkl_tensor_t`
  - `right_tensor`: input tensor, shape `[N_COL, N_INNER]`, layout and type defined by `sdkl_tensor_t`

  @param[in]  platform       Execution platform (CPU, NPU0, NPU1, GPU) for SDKL operation.
  @param[out] result_tensor  Pointer to the output tensor descriptor.
  @param[in]  left_tensor    Pointer to the left-hand input tensor descriptor.
  @param[in]  right_tensor   Pointer to the right-hand input tensor descriptor (typically transposed weights).

  @return
  - `AEE_SUCCESS` on successful execution.
  - Error codes from `AEEStdErr.h` (e.g., `AEE_EBADPARM`, `AEE_EFAILED`) if validation or execution fails.
*/
int sdkl_mm_tensor(
  sdkl_tensor_platform_e platform,
  sdkl_tensor_t* restrict result_tensor,
  const sdkl_tensor_t* restrict left_tensor,
  const sdkl_tensor_t* restrict right_tensor
);

/*!
  @ingroup CPUMacroMatMul
  @brief
  Performs matrix multiplication of FP16 activations by FP16 weights, producing FP16 results.

  Hexagon/HMX-native memory layouts. It avoids data layout and type conversion overhead, assuming the caller
  provides inputs in the expected formats:
  - `A`: output matrix, type FP16, layout AH (activation layout for the HMX unit)
  - `X`: input matrix, type FP16, layout AH (activation layout for the HMX unit)
  - `W`: weight matrix, type FP16, layout WH (weight layout for the HMX unit)

  @param[in]  domain     Which compute DSP (CDSP) core the SDKL library will interact with.
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
int sdkl_npu_mm_f16(int domain, int n_row, int n_col, int n_inner, _Float16* A, const _Float16* X, const _Float16* W);

/*!
  @ingroup CPUMacroMatMul
  @brief
  Performs matrix multiplication of FP32 activations by FP16 weights, producing FP32 results.

  This kernel is optimized for mixed-precision matmul on the Hexagon NPU. It assumes that:
  - `A`: output matrix, type FP32, in standard row-major CPU layout.
  - `X`: input matrix, type FP32, in standard row-major CPU layout.
  - `W`: weight matrix, type FP16, pre-layouted in WH format (HMX unit layout).

  The function avoids layout and type conversion overhead for weights by requiring them
  to be preprocessed using `sdkl_cpu_rm_to_wh_f16_inplace()`.

  @param[in]  domain     Which compute DSP (CDSP) core the SDKL library will interact with.
  @param[in]  n_row      Number of rows in matrix X and A.
  @param[in]  n_col      Number of columns in matrix W and A.
  @param[in]  n_inner    Shared dimension between X and W (columns of X, rows of W).
  @param[out] A          Pointer to the output matrix A (FP32, row-major layout).
  @param[in]  X          Pointer to the input matrix X (FP32, row-major layout).
  @param[in]  W          Pointer to the weight matrix W (FP16, WH layout).

  @return
  - `AEE_SUCCESS` on successful execution.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  The input and output buffers must be allocated using `sdkl_npu_alloc()` to ensure proper alignment
  and compatibility with the NPU. The weight matrix must be in SDKL's WH format.
 */
int sdkl_npu_mm_f32f16_f32(int domain, int n_row, int n_col, int n_inner, float* A, const float* X, const _Float16* W);

/*!
  @ingroup CPUMacroMatMul
  @brief
  Performs matrix multiplication of FP16 activations by FP16 weights, producing FP16 results.

  This kernel is optimized for FP16 matmul on the Hexagon NPU. It assumes that:
  - `A`: output matrix, type FP16, in standard row-major CPU layout.
  - `X`: input matrix, type FP16, in standard row-major CPU layout.
  - `W`: weight matrix, type FP16, pre-layouted in WH format (HMX unit layout).

  The function avoids layout and type conversion overhead for weights by requiring them
  to be preprocessed using `sdkl_cpu_f16_rm_to_f16_wh_inplace()`.

  @param[in]  domain     Which compute DSP (CDSP) core the SDKL library will interact with.
  @param[in]  n_row      Number of rows in matrix X and A.
  @param[in]  n_col      Number of columns in matrix W and A.
  @param[in]  n_inner    Shared dimension between X and W (columns of X, rows of W).
  @param[out] A          Pointer to the output matrix A (FP16, row-major layout).
  @param[in]  X          Pointer to the input matrix X (FP16, row-major layout).
  @param[in]  W          Pointer to the weight matrix W (FP16, WH layout).

  @return
  - `AEE_SUCCESS` on successful execution.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  The input and output buffers must be allocated using `sdkl_npu_alloc()` to ensure proper alignment
  and compatibility with the NPU. The weight matrix must be in SDKL's WH format.
 */
int sdkl_npu_mm_f16f16_f16(
  int domain,
  int n_row,
  int n_col,
  int n_inner,
  _Float16* A,
  const _Float16* X,
  const _Float16* W
);

/*!
  @ingroup CPUMacroMatMul
  @brief
  Performs matrix multiplication of ui8 activations by i8 weights, producing i32 results.

  This kernel is optimized for i32 matmul on the Hexagon NPU. It assumes that:
  - `A`                   : output matrix, type i32, in standard row-major CPU layout.
  - `X`                   : input matrix, type ui8, in standard row-major CPU layout.
  - `W`                   : weight matrix, type i8, pre-layouted in WH format (HMX unit layout).

  The function avoids layout and type conversion overhead for weights by requiring them
  to be preprocessed using `sdkl_cpu_rm_to_wh_i8()`.

  @param[in]  domain              Which compute DSP (CDSP) core the SDKL library will interact with.
  @param[in]  n_row               Number of rows in matrix X and A.
  @param[in]  n_col               Number of columns in matrix W and A.
  @param[in]  n_inner             Shared dimension between X and W (columns of X, rows of W).
  @param[out] A                   Pointer to the output matrix A (i32, row-major layout).
  @param[in]  X                   Pointer to the input matrix X (ui8, row-major layout).
  @param[in]  W                   Pointer to the weight matrix W (i8, WH layout).

  @return
  - `AEE_SUCCESS` on successful execution.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  The input and output buffers must be allocated using `sdkl_npu_alloc()` to ensure proper alignment
  and compatibility with the NPU. The weight matrix must be in SDKL's WH format.
 */
int sdkl_npu_mm_u8i8_i32(int domain, int n_row, int n_col, int n_inner, int32_t* A, const uint8_t* X, const int8_t* W);

/*!
  @ingroup CPUMacroMatMul
  @brief
  Performs matrix multiplication of ui8 activations by i4 weights, producing i32 results.

  This kernel is optimized for i32 matmul on the Hexagon NPU. It assumes that:
  - `A`                   : output matrix, type i32, in standard row-major CPU layout.
  - `X`                   : input matrix, type ui8, in standard row-major CPU layout.
  - `W`                   : weight matrix, type i4, in WH format (HMX unit layout).

  The function avoids layout and type conversion overhead for weights by requiring them
  to be preprocessed using `sdkl_cpu_rm_to_wh_i4()`.

  @param[in]  domain              Which compute DSP (CDSP) core the SDKL library will interact with.
  @param[in]  n_row               Number of rows in matrix X and A.
  @param[in]  n_col               Number of columns in matrix W and A.
  @param[in]  n_inner             Shared dimension between X and W (columns of X, rows of W).
  @param[out] A                   Pointer to the output matrix A (i32, row-major layout).
  @param[in]  X                   Pointer to the input matrix X (ui8, row-major layout).
  @param[in]  W                   Pointer to the weight matrix W (i4, WH layout).

  @return
  - `AEE_SUCCESS` on successful execution.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  The input and output buffers must be allocated using `sdkl_npu_alloc()` to ensure proper alignment
  and compatibility with the NPU. The weight matrix must be in SDKL's WH format.
 */
int sdkl_npu_mm_u8i4_i32(
  int domain,
  size_t n_row,
  size_t n_col,
  size_t n_inner,
  int32_t* A,
  const uint8_t* X,
  const uint8_t* W
);

/*!
  @ingroup HexKLCPUMacro
  @defgroup CPUMacroLayout Data Layout Transformation Functions
  @brief Defines functions for transforming between data layouts including row-major and HMX optimized formats.
*/
/*!
  @ingroup CPUMacroLayout
  @brief Applies the AH (activation layout for HMX unit) data layout to input activations in-place.

  This function transforms the input matrix `X` from row-major (RM) layout
  to the AH layout used by the HMX unit. The transformation is performed in-place,
  modifying the contents of `X` directly.

  The AH layout is optimized for NPU execution and must be applied before passing
  activations to HMX-based operators. The input buffer must be allocated using
  `sdkl_npu_alloc()` to ensure proper alignment and compatibility with
  HMX unit's memory requirements.

  @param[in]     n_row Number of rows in the input matrix.
  @param[in]     n_col Number of columns in the input matrix.
  @param[in,out] X Pointer to the input matrix stored in row-major order.
           The matrix is expected to be of size `n_row * n_col` and contain `_Float16` values.

  @return
  - `AEE_SUCCESS` on successful transformation.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  This function assumes that the input buffer `X` is mutable and properly aligned.
 */
int sdkl_cpu_rm_to_ah_f16_inplace(size_t n_row, size_t n_col, _Float16* X);

/*!
  @ingroup CPUMacroLayout
  @brief Applies the WH (weight layout for HMX unit) data layout to weights in-place.

  This function transforms the input weight matrix `W` from row-major (RM) layout
  to the WH layout required by the HMX unit. The transformation is performed
  in-place and prepares the weights for optimized execution on the NPU.

  @param[in]     n_row Number of rows in the input weight matrix.
  @param[in]     n_col Number of columns in the input weight matrix.
  @param[in,out] W Pointer to the input weight matrix stored in row-major order.
           The matrix is expected to be of size `n_row * n_col` and contain `_Float16` values.

  @return
  - `AEE_SUCCESS` on successful transformation.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  This function assumes that the input buffer `W` is mutable and properly aligned.
 */
int sdkl_cpu_rm_to_wh_f16_inplace(size_t n_row, size_t n_col, _Float16* W);

/*!
  @ingroup CPUMacroLayout
  @brief Applies the WH (weight layout for HMX unit) data layout to weights in-place.

  This function transforms the input weight matrix `W` from row-major (RM) layout
  to the WH layout required by the HMX unit. The transformation is performed
  in-place and prepares the weights for optimized execution on the NPU.

  @param n_row Number of rows in the input matrix.
  @param n_col Number of columns in the input matrix.
  @param W Pointer to the input matrix stored in row-major order.
           The matrix is expected to be of size `n_row * n_col` and contain `int8_t` values.

  @return
  - `AEE_SUCCESS` on successful transformation.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  This function assumes that the input buffer `W` is mutable and properly aligned.
 */
int sdkl_cpu_rm_to_wh_i8_inplace(size_t n_row, size_t n_col, int8_t* W);

/*!
  @ingroup CPUMacroLayout
  @brief Converts a row-major matrix to WH layout for HMX unit.

  This function transforms the input matrix `X_i8_cpu`, stored in standard row-major format,
  into the WH layout required by the HMX unit.

  @param[in]  n_inner Number of columns in the input matrix.
  @param[in]  n_row   Number of rows in the input matrix.
  @param[in]  X_i8_cpu Pointer to the input matrix stored in row-major order.
                  The matrix is expected to be of size `n_row * n_inner` and contain `uint8_t` values.
  @param[out] Xq Pointer to the output buffer where the WH-formatted matrix will be stored.
            The buffer must be preallocated and large enough to hold the transformed data.

  @return
  - `AEE_SUCCESS` on successful transformation.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  This function assumes that both input and output buffers are valid and properly aligned.
 */
int sdkl_cpu_rm_to_wh_i8(size_t n_inner, size_t n_row, const uint8_t* X_i8_cpu, uint8_t* Xq);

/*!
  @ingroup CPUMacroLayout
  @brief Converts matrix multiplication output from HMX layout to standard row-major layout (ui8i8 variant).

  This function transforms the output matrix from ui8i8 matrix multiplication operations
  from the HMX activation layout back to standard row-major CPU layout.

  This function is typically called after NPU ui8i8 matrix multiplication operations to
  convert results back to a format suitable for CPU processing.

  @param[in]  n_row      Number of rows in the matrix.
  @param[in]  n_col      Number of columns in the matrix.
  @param[in]  A_i32_tmp  Pointer to the input matrix in HMX activation layout.
                         The matrix contains i32 values organized in 64×32 blocks.
  @param[out] A          Pointer to the output matrix in standard row-major layout.
                         The buffer must be preallocated with size `n_row * n_col * sizeof(int32_t)`.

  @return
  - `AEE_SUCCESS` on successful transformation.
  - `AEE_EBADCLASS` if either input pointer is NULL.
  - Error codes defined in `AEEStdErr.h` in case of other failures.

  @note
  Both input and output buffers must be properly allocated and aligned.
  The function assumes the input matrix follows the exact HMX activation layout
  as produced by NPU ui8i8 matrix multiplication operations.

  @warning
  The caller is responsible for ensuring that both buffers have sufficient space
  for the matrix data. Buffer overruns will result in undefined behavior.
 */
int sdkl_cpu_ui8i8_ah_to_i32_rm(size_t n_row, size_t n_col, int32_t* A_i32_tmp, int32_t* A);

/*!
  @ingroup CPUMacroLayout
  @brief Converts matrix multiplication output from HMX layout to standard row-major layout (ui8i4 variant).

  This function transforms the output matrix from ui8i4 matrix multiplication operations
  from the HMX back to standard row-major CPU layout.

  This function is typically called after NPU ui8i4 matrix multiplication operations to
  convert results back to a format suitable for CPU processing.

  @param[in]  n_row      Number of rows in the matrix.
  @param[in]  n_col      Number of columns in the matrix.
  @param[in]  A_i32_hmx  Pointer to the input matrix in HMX.
                         The matrix contains i32 values organized in 64×32 blocks.
  @param[out] A_i32_cpu  Pointer to the output matrix in standard row-major layout.
                         The buffer must be preallocated with size `n_row * n_col * sizeof(int32_t)`.

  @return
  - `AEE_SUCCESS` on successful transformation.
  - `AEE_EBADCLASS` if either input pointer is NULL.
  - Error codes defined in `AEEStdErr.h` in case of other failures.

  @note
  Both input and output buffers must be properly allocated and aligned.
  The function assumes the input matrix follows the exact HMX format
  as produced by NPU ui8i4 matrix multiplication operations.

  @warning
  The caller is responsible for ensuring that both buffers have sufficient space
  for the matrix data. Buffer overruns will result in undefined behavior.
 */
int sdkl_cpu_ui8i4_ah_to_i32_rm(size_t n_row, size_t n_col, int32_t* A_i32_hmx, int32_t* A_i32_cpu);

/*!
  @ingroup CPUMacroLayout
  @brief Recovers the row-major layout from AH (activation layout for HMX unit) in-place.

  This function transforms the input matrix `A` from the AH layout
  back to standard row-major (RM) layout. The transformation is performed in-place,
  reversing the spatial reorganization applied for NPU execution.

  @param[in]      n_row Number of rows in the activation matrix.
  @param[in]      n_col Number of columns in the activation matrix.
  @param[in,out]  A Pointer to the activation matrix stored in AH layout.
           The matrix is expected to be of size `n_row * n_col` and contain `_Float16` values.

  @return
  - `AEE_SUCCESS` on successful transformation.
  - Error codes defined in `AEEStdErr.h` (e.g., `AEE_EFAILED`, `AEE_EBADPARM`, etc.) in case of failure.

  @note
  This function assumes that the input buffer `A` is mutable and properly aligned.
 */
int sdkl_cpu_ah_to_rm_f16_inplace(size_t n_row, size_t n_col, _Float16* A);

/*!
  @ingroup CPUMacroLayout
  @brief Converts a row-major i4 matrix to WH layout for HMX unit.

  This function transforms an input matrix from standard row-major layout containing
  i4 values (stored as int8_t with one i4 value per byte, sign-extended) to the
  specialized WH (Weight-Hexagon) layout required by the NPU matrix multiplication unit.

  The input i4 format expects:
  - Each i4 value is stored in a separate int8_t byte
  - Values are sign-extended: valid range is -8 to +7
  - Only the lower 4 bits contain the actual i4 value
  - Values are arranged in standard row-major order

  @param[out] full_wt_tiled Pointer to output buffer in WH layout.
                            Size must be at least `((wt_rows + 31) / 32) * ((wt_cols + 31) / 32) * 512` bytes.
  @param[in]  wt_old        Pointer to input matrix in row-major layout.
                            Each element contains one i4 value stored as int8_t (sign-extended).
  @param[in]  wt_rows       Number of rows in input matrix.
  @param[in]  wt_cols       Number of columns in input matrix.

  @return AEE_SUCCESS on success, error code on failure

  @note
  - Input values must be in the range [-8, +7] (valid i4 range)
  - Output buffer must be properly aligned and sized for the tiled format
  - The transformation is optimized for NPU hardware
 */
int sdkl_cpu_rm_to_wh_i4(uint8_t* full_wt_tiled, int8_t* wt_old, size_t wt_rows, size_t wt_cols);

/*!
  @ingroup HexKLCPUMacro
  @defgroup CPUMacroValidate Validation Functions
  @brief Defines functions for validating tensors and tensor operations.
*/
/*!
  @ingroup CPUMacroValidate
  @brief Validates the integrity and consistency of a tensor descriptor.

  This function performs a comprehensive validation of the provided `sdkl_tensor_t` structure.
  It checks for:
  - Null pointers
  - Valid number of dimensions (`ndims`)
  - Non-zero dimensions and strides
  - Proper memory bounds based on `dims`, `strides`, and `data_offset`
  - Contiguity and transposition consistency for 1D and 2D tensors
  - Valid data types and quantization modes
  - Compatibility between `data_dtype` and `quantization`

  @param[in] tensor Pointer to the tensor descriptor to validate.

  @return AEE_SUCCESS (0) if the tensor is valid.
          AEE_EBADCLASS if the tensor or its data pointer is NULL.
          AEE_EBADPARM if any field is invalid or inconsistent.

  @note This function assumes that the tensor descriptor is fully initialized.
        It does not perform memory allocation or deallocation.

  @warning If `is_continuous` is set, the function expects strides to match
           a contiguous layout (row-major or transposed) for 1D and 2D tensors.
 */
int sdkl_tensor_validate(const sdkl_tensor_t* tensor);

/*!
  @ingroup CPUMacroValidate
  @brief Validates that two tensors can be multiplied and the result tensor is correctly configured.

  Supports only 1D and 2D tensors. Assumes all tensors are individually validated.
  Enforces:
  - Shape compatibility for multiplication
  - Only right tensor may be quantized

  @param[in] result_tensor Pointer to the output tensor descriptor (must be preconfigured).
  @param[in] left_tensor   Pointer to the left-hand-side tensor.
  @param[in] right_tensor  Pointer to the right-hand-side tensor.

  @return AEE_SUCCESS if all tensors are valid and compatible.
          AEE_EBADCLASS if any tensor pointer is NULL.
          AEE_EBADPARM if tensor ranks, shapes, or metadata are incompatible.
 */
int sdkl_mm_tensor_validate(
  const sdkl_tensor_t* restrict result_tensor,
  const sdkl_tensor_t* restrict left_tensor,
  const sdkl_tensor_t* restrict right_tensor
);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //__SDKL_H__
