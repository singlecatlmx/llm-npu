#!/bin/bash
#===============================================================================
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#===============================================================================

# Default values
HEX_ARCH="v73"
ARM_ARCH="armv8"
CPU_OS="android26"

# Help message
print_help() {
  echo "Usage: $0 [--hex-arch <v73|v75|v79>] [--arm-arch <armv8|armv9>] [--cpu-os <android26|qclinux>] [--help]"
  echo ""
  echo "Options:"
  echo "  --hex-arch   Set Hexagon architecture version (default: v73)"
  echo "  --arm-arch   Set ARM architecture version (default: armv8)"
  echo "  --cpu-os     Set CPU OS (default: android26). Note: qclinux supported only with armv8"
  echo "  --help       Show this help message"
  exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hex-arch)
      HEX_ARCH="$2"
      shift 2
      ;;
    --arm-arch)
      ARM_ARCH="$2"
      shift 2
      ;;
    --cpu-os)
      CPU_OS="$2"
      shift 2
      ;;
    --help)
      print_help
      ;;
    *)
      echo "Unknown option: $1"
      print_help
      ;;
  esac
done

# Validate HEX_ARCH
if [[ "$HEX_ARCH" != "v73" && "$HEX_ARCH" != "v75" && "$HEX_ARCH" != "v79" ]]; then
  echo "Error: Unsupported hex_arch '$HEX_ARCH'"
  print_help
fi

# Validate ARM_ARCH
if [[ "$ARM_ARCH" != "armv8" && "$ARM_ARCH" != "armv9" ]]; then
  echo "Error: Unsupported arm_arch '$ARM_ARCH'"
  print_help
fi

# Validate CPU_OS
if [[ "$CPU_OS" != "android26" && "$CPU_OS" != "qclinux" ]]; then
  echo "Error: Unsupported cpu_os '$CPU_OS'"
  print_help
fi

# Enforce compatibility
if [[ "$ARM_ARCH" == "armv9" && "$CPU_OS" == "qclinux" ]]; then
  echo "Error: qclinux is only supported with armv8 architecture."
  print_help
fi

# Check required environment variables
if [ -z "$DEFAULT_HEXAGON_TOOLS_ROOT" ]; then
  echo "Error: DEFAULT_HEXAGON_TOOLS_ROOT is not set."
  exit 1
fi

if [ -z "$DEFAULT_TOOLS_VARIANT" ]; then
  echo "Error: DEFAULT_TOOLS_VARIANT is not set."
  exit 1
fi

if [ -z "$ADB_FLAGS" ]; then
  echo "Error: ADB_FLAGS is not set."
  exit 1
fi

# Extract algorithm name from parent directory
ALGO_NAME=$(basename "$(dirname "$(realpath "$0")")")

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_HEXKL="${SCRIPT_DIR}/../../lib/hexagon_${DEFAULT_TOOLS_VARIANT}_${HEX_ARCH}/libhexkl_skel.so"
LIB_SDKL="${SCRIPT_DIR}/../../lib/${ARM_ARCH}_${CPU_OS}/libsdkl.so"
TEST_BIN="${SCRIPT_DIR}/build/${ARM_ARCH}_${CPU_OS}/test_${ALGO_NAME}"

# Check required files
if [[ ! -f "$LIB_HEXKL" ]]; then
  echo "Error: $LIB_HEXKL not found."
  exit 1
fi

if [[ ! -f "$LIB_SDKL" ]]; then
  echo "Error: $LIB_SDKL not found."
  exit 1
fi

if [[ ! -f "$TEST_BIN" ]]; then
  echo "Error: $TEST_BIN not found. Did you run build.sh?"
  exit 1
fi

# Run commands
echo "Using Hexagon architecture: $HEX_ARCH"
echo "Using ARM architecture: $ARM_ARCH"
echo "Using CPU OS: $CPU_OS"

adb $ADB_FLAGS push "$TEST_BIN" /data/local/tmp/
adb $ADB_FLAGS push "$LIB_SDKL" /data/local/tmp/
adb $ADB_FLAGS push "$LIB_HEXKL" /data/local/tmp/
adb $ADB_FLAGS shell "cd /data/local/tmp; ADSP_LIBRARY_PATH=/data/local/tmp LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/test_$ALGO_NAME"