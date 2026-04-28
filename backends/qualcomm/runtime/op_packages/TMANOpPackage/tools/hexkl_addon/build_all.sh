#!/bin/bash
#===============================================================================
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#===============================================================================

set -x

bash "examples/sdkl_npu_mm_f32f16_f32/build.sh" --arm-arch armv8 --cpu-os android26

bash "examples/sdkl_npu_mm_f32f16_f32/build.sh" --arm-arch armv8 --cpu-os qclinux

bash "examples/sdkl_npu_mm_f32f16_f32/build.sh" --arm-arch armv9 --cpu-os android26

bash "examples/sdkl_npu_mm_f16f16_f16/build.sh" --arm-arch armv8 --cpu-os android26

bash "examples/sdkl_npu_mm_f16f16_f16/build.sh" --arm-arch armv8 --cpu-os qclinux

bash "examples/sdkl_npu_mm_f16f16_f16/build.sh" --arm-arch armv9 --cpu-os android26

bash "examples/sdkl_mm_tensor/build.sh" --arm-arch armv8 --cpu-os android26

bash "examples/sdkl_mm_tensor/build.sh" --arm-arch armv8 --cpu-os qclinux

bash "examples/sdkl_mm_tensor/build.sh" --arm-arch armv9 --cpu-os android26

bash "examples/sdkl_npu_mm_u8i4_i32/build.sh" --arm-arch armv8 --cpu-os android26

bash "examples/sdkl_npu_mm_u8i4_i32/build.sh" --arm-arch armv8 --cpu-os qclinux

bash "examples/sdkl_npu_mm_u8i4_i32/build.sh" --arm-arch armv9 --cpu-os android26

bash "examples/sdkl_npu_mm_u8i8_i32/build.sh" --arm-arch armv8 --cpu-os android26

bash "examples/sdkl_npu_mm_u8i8_i32/build.sh" --arm-arch armv8 --cpu-os qclinux

bash "examples/sdkl_npu_mm_u8i8_i32/build.sh" --arm-arch armv9 --cpu-os android26

bash "examples/hexkl_micro_hmx_mm_u8i4_i32/build.sh" --hex-arch v73

bash "examples/hexkl_micro_hmx_mm_u8i4_i32/build.sh" --hex-arch v75

bash "examples/hexkl_micro_hmx_mm_u8i4_i32/build.sh" --hex-arch v79

bash "examples/hexkl_micro_hmx_mm_u8i8_i32/build.sh" --hex-arch v73

bash "examples/hexkl_micro_hmx_mm_u8i8_i32/build.sh" --hex-arch v75

bash "examples/hexkl_micro_hmx_mm_u8i8_i32/build.sh" --hex-arch v79

bash "examples/hexkl_micro_hmx_mm_f16/build.sh" --hex-arch v73

bash "examples/hexkl_micro_hmx_mm_f16/build.sh" --hex-arch v75

bash "examples/hexkl_micro_hmx_mm_f16/build.sh" --hex-arch v79

bash "examples/hexkl_macro_mm_f16/build.sh" --hex-arch v73

bash "examples/hexkl_macro_mm_f16/build.sh" --hex-arch v75

bash "examples/hexkl_macro_mm_f16/build.sh" --hex-arch v79

