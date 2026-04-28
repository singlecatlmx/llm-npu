Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

Simple Test for `hexkl_micro.a` API: `test_hexkl_micro_hmx_mm_f16`

Overview
--------
This project provides a minimal test harness for the `hexkl_micro.a` library, specifically targeting matrix multiplication using the HMX engine via the micro API.

**Note:** This harness is intended to be executed on the Hexagon simulator environment. 


The test demonstrates usage of the following API functions:

- int hexkl_micro_get_version
- int hexkl_micro_hw_init
- int hexkl_micro_hmx_lock
- int hexkl_micro_hmx_unlock
- int hexkl_micro_hmx_config_size
- int hexkl_micro_hmx_setup_acc_read_f16
- int hexkl_micro_hmx_copy_submatrix_to_f16
- int hexkl_micro_hmx_rm_to_ah_f16
- int hexkl_micro_hmx_acc_clear_f16
- int hexkl_micro_hmx_rm_to_wh_f16
- int hexkl_micro_hmx_mm_f16
- int hexkl_micro_hmx_acc_read_f16
- int hexkl_micro_hmx_ah_to_rm_f16
- int hexkl_micro_hmx_copy_f16_to_f32_submatrix

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
  ./build.sh --hex-arch <v73|v75|v79>;

Options:
  --hex-arch <v73|v75|v79>;   Specifies the Hexagon architecture version. Default is v73.
  --help                     Displays usage information.

The compiled output is placed in:
  hexagon_<DEFAULT_TOOLS_VARIANT>_<v73|v75|v79>

run_simulator.sh

Runs the compiled binary using the Hexagon simulator.

Usage:
  ./run_simulator.sh --help
  ./run_simulator.sh --hex-arch <v73|v75|v79>;

Options:
  --hex-arch <v73|v75|v79>;   Specifies the Hexagon architecture version to run. Default is v73.
  --help                     Displays usage information.

The simulator loads the binary and configuration files from:
  hexagon_<DEFAULT_TOOLS_VARIANT>_<v73|v75|v79>

Notes
-----
- This example is distributed as-is and does not use a Makefile. It is intended for demonstration and testing only.
- It depends on the Hexagon SDK to be installed and properly configured.
- NPU programmers may adapt the initialization and locking routines to suit their own application needs.

Linkage with `libhexkl_micro.a`
------------------------------
The build process links user-defined object files with the `libhexkl_micro.a` static library to create a shared NPU library compatible with the Hexagon simulator. The linker command in `build.sh` uses the Hexagon toolchain and includes architecture-specific flags, memory wrappers, and shared object generation options. 

        -m${HEX_ARCH} -G0 -fpic -Wl,-Bsymbolic \
        -Wl,-L$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/target/hexagon/lib/${HEX_ARCH}/G0/pic \
        -Wl,-L$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/target/hexagon/lib/ \
        -Wl,--no-threads -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -shared \
        -o $EXE_BUILD_DIR/$SO_NAME -Wl,-soname,$SO_NAME \
        -Wl,--start-group $EXE_BUILD_DIR/$OBJ_FILE \
         $SCRIPT_DIR/../../lib/$BUILD_DIR/libhexkl_micro.a -Wl,--end-group -lc

Users must ensure that:

- `${HEX_ARCH}` is set to the correct target (`v73`, `v75`, or `v79`).
- `$DEFAULT_HEXAGON_TOOLS_ROOT` is initialized by sourcing the Hexagon SDK setup script.
- `$EXE_BUILD_DIR` points to the desired output directory.
- `$OBJ_FILE` contains the list of custom object files.
- The path to libhexkl_micro.a is correctly set using the $SCRIPT_DIR variable, 
  e.g., $SCRIPT_DIR/../../lib/hexagon_toolv88_v75/libhexkl_micro.a for v75..

The linker command includes the following switches:

- `-m${HEX_ARCH}`: Specifies the Hexagon architecture.
- `-G0`: Uses the small data section for performance.
- `-fpic`: Generates position-independent code for shared libraries.
- `-Wl,-Bsymbolic`: Resolves symbols at link time to avoid runtime conflicts.
- `-Wl,-L<path>`: Adds library search paths.
- `-Wl,--no-threads`: Disables multi-threaded linking.
- `--wrap=malloc`, `--wrap=calloc`, etc.: Redirects memory functions to custom wrappers.
- `-shared`: Produces a shared object.
- `-Wl,-soname,<name>`: Sets the shared object name.
- `-Wl,--start-group ... -Wl,--end-group`: Ensures all symbols are resolved.
- `-lc`: Links the standard C library.

This setup ensures proper symbol resolution and compatibility with the Hexagon simulator runtime.

Output
------
Upon successful execution, the simulator will produce performance statistics in:

  hexagon_<DEFAULT_TOOLS_VARIANT>_<arch>/pmu_stats.txt
