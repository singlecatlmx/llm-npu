Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

Simple Test for `hexkl_micro.a` API: `hexkl_micro_hmx_mm_ui8i8_i32`

Overview
--------
This project provides a minimal test harness for the `hexkl_micro.a` library, specifically targeting matrix multiplication using the HMX engine via the micro API.

> **Note:** This harness is intended to be executed on the Hexagon simulator environment. 

The test demonstrates usage of the following API functions:

  int hexkl_micro_get_version
  int hexkl_micro_hw_init
  int hexkl_micro_hmx_lock
  int hexkl_micro_hmx_unlock
  int hexkl_micro_hmx_config_size
  int hexkl_micro_hmx_setup_acc_read_int32
  int hexkl_micro_hmx_copy_submatrix_to_8b_activation
  int hexkl_micro_hmx_acc_clear_int32
  int hexkl_micro_hmx_rm_to_wh_i8
  int hexkl_micro_hmx_mm_u8i8
  int hexkl_micro_hmx_acc_read_int32
  int hexkl_micro_hmx_copy_32b_to_submatrix

These functions collectively demonstrate initialization, memory layout conversion, matrix multiplication, and result retrieval using the HMX engine.

Prerequisites
-------------
1. Hexagon SDK Environment

You must source the Hexagon SDK setup script to configure necessary environment variables:

  source $HEXAGON_SDK_ROOT/setup_sdk_env.source

If this step is skipped, the build.sh script will fail due to missing environment variables.

Scripts
-------
build.sh

Compiles the test binary using the Hexagon SDK. Make sure the SDK environment is sourced before running.

Usage:
  ./build.sh --help
  ./build.sh --hex-arch <v73|v75|v79>

Options:
  --hex-arch <v73|v75|v79>   Specifies the Hexagon architecture version. Default is v73.
  --help                     Displays usage information.

The compiled output is placed in:
  hexagon_<DEFAULT_TOOLS_VARIANT>_<v73|v75|v79>

run_simulator.sh

Runs the compiled binary using the Hexagon simulator.

Usage:
  ./run_simulator.sh --help
  ./run_simulator.sh --hex-arch <v73|v75|v79>

Options:
  --hex-arch <v73|v75|v79>   Specifies the Hexagon architecture version to run. Default is v73.
  --help                     Displays usage information.

The simulator loads the binary and configuration files from:
  hexagon_<DEFAULT_TOOLS_VARIANT>_<v73|v75|v79>

Notes
-----
- This example is distributed as-is and does not use a Makefile. It is intended for demonstration and testing only.
- It depends on the Hexagon SDK to be installed and properly configured.
- NPU programmers may adapt the initialization and locking routines to suit their own application needs.

Output
------
Upon successful execution, the simulator will produce performance statistics in:

  hexagon_<DEFAULT_TOOLS_VARIANT>_<arch>/pmu_stats.txt
