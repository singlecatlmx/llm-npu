
Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

# Simple Test for `libsdkl.so` API: `sdkl_npu_mm_f16f16_f16`

## Overview

This project provides a minimal test harness for the `libsdkl.so` library, specifically targeting the API:

```c
int sdkl_npu_mm_f16f16_f16(int domain, 
  int n_row,
  int n_col,
  int n_inner,
  _Float16* A,
  const _Float16* X,
  const _Float16* W
);
```

This function performs matrix multiplication with mixed precision inputs and outputs.

## Prerequisites

### 1. Hexagon SDK Environment

You **must** source the Hexagon SDK setup script to configure necessary environment variables:

```bash
source $SDK_HOME/setup_sdk_env.source
```

If this step is skipped, the `build.sh` script will **fail** due to missing environment variables.

### 2. Android Device Configuration

The `run_android.sh` script requires manual setup of the following environment variable:

- `ADB_FLAGS`: ADB flags that will be in use.

Example in case you are using a remote remote android device:

```bash
export ADB_FLAGS=-H /path/to/android/host -s your_device_serial
```

Example in case you are using local android device:

```bash
export ADB_FLAGS=-s your_device_serial
```

## Scripts

### `build.sh`

Compiles the test binary using the Hexagon SDK. Make sure the SDK environment is sourced before running.

```bash
./build.sh --help
./build.sh --arm-arch <armv8|armv9>
```

### `run_android.sh`

Deploys and runs the test on an Android device or QC Linux target. It supports the following options:

```bash
./run_android.sh --help
./run_android.sh --hex-arch <v73|v75|v79>
./run_android.sh --arm-arch <armv8|armv9>
./run_android.sh --cpu-os <android26|qclinux>
```

- The `--hex-arch` switch determines which precompiled `libhexkl_skel.so` to load onto the device. The library is loaded from:
  ```
  ../../lib/hexagon_<DEFAULT_TOOLS_VARIANT>_<v73|v75|v79>
  e.g: ../../lib/hexagon_toolv88_v75 in case of hexagon tools 8.8.06 and v75
  ```

- The `--arm-arch` switch determines which precompiled `libsdkl.so` to load. The library is loaded from:
  ```
  ../../lib/<armv8|armv9>_<cpu-os>
  e.g: ../../lib/armv8_android26 or ../../lib/armv8_qclinux
  ```

- The `--cpu-os` switch selects the target operating system for the CPU side. Supported values are:
  - `android26`: for Android-based deployment
  - `qclinux`: for QC Linux-based deployment (only supported with `armv8`)

This switch affects both the location of the `libsdkl.so` and the test binary that gets pushed to the device.
```
