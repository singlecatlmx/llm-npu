Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

# hexkl_addon Package

This package contains components of the HexKL API for deployment and testing.

## Contents

### 1. `lib/`
- Contains architecture- and OS-specific builds of the HexKL API.
- `armv8_android26/`, `armv9_android26/`, and `armv8_qclinux/`:
  - `libsdkl.so`: HexKL CPU Macro API, invoked by ARM CPU applications.
- `hexagon_toolv<"major minor tools version">_<v73/v75/v79>/`:
  - `libhexkl_skel.so` : FastRPC layer between CPU and NPU.
  - `libhexkl_macro.a` : HexKL NPU Macro API, invoked by CPU Macro API via FastRPC or used directly by NPU programmers.
  - `libhexkl_macro.so`: HexKL NPU Macro API, dynamic library similar to static `libhexkl_macro.a`. To be used directly by NPU programmers.
  - `libhexkl_micro.a` : HexKL NPU Micro API, providing low-level single-thread routines for NPU programmers.

### 2. `include/`
- `sdkl.h`       : Header for HexKL CPU Macro API.
- `hexkl_macro.h`: Header for HexKL NPU Macro API.
- `hexkl_micro.h`: Header for HexKL NPU Micro API.

### 3. `examples/`
- Contains example projects demonstrating usage of HexKL CPU Macro API, HexKL Macro API, and HexKL Micro API.
- Each example includes a README with build and execution instructions.

### 4. `build_all.sh`
- Script to build all examples with appropriate Hexagon and ARM architecture flags.
- Supports `--arm-arch` and `--cpu-os` switches to build for multiple targets.

### 5. `run_all.sh`
- Script to execute all examples.
- Automatically runs `run_simulator.sh` and `run_android.sh` if present in each example.
- Logs all output to `run_all_log.txt`.
- Prefixes all script messages with `[RUN_ALL_SCRIPT]` for clarity.
- Saves a final summary report to `run_all_final_report.txt`.

#### Supported switches:
- `--hex-arch <v73|v75|v79>`: Specify Hexagon architecture version (default: v73).
- `--arm-arch <armv8|armv9>`: Specify ARM architecture version (default: armv8).
- `--cpu-os <android26|qclinux>`: Specify CPU OS target (default: android26). Note: `qclinux` is only supported with `armv8`.
- `--adb-flags <string>`: Set ADB_FLAGS environment variable for Android execution.
  - You can export `ADB_FLAGS` manually before running `run_all.sh`:
    ```bash
    export ADB_FLAGS="<your_flags_here>"
    ./run_all.sh
    ```
  - If both are used, the `--adb-flags` switch will override any previously exported `ADB_FLAGS`.

---
