#!/bin/bash
#===============================================================================
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#===============================================================================

print_help() {
  echo "Usage: $0 [--arm-arch <armv8|armv9>] [--help]"
  echo ""
  echo "Options:"
  echo "  --arm-arch <armv8|armv9>       Specify ARM architecture version (default: armv8)"
  echo "  --cpu-os <android26|qclinux>   Specify CPU OS (default: android26). Note: qclinux supported for armv8 only"
  echo "  --help                         Show this help message"
}

# Default ARM architecture
ARM_ARCH="armv8"

#Default CPU OS
CPU_OS="android26"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm-arch)
      shift
      if [[ "$1" =~ ^armv8$|^armv9$ ]]; then
        ARM_ARCH="$1"
      else
        echo "Error: Unsupported ARM architecture '$1'"
        print_help
        exit 1
      fi
      ;;
    --cpu-os)
      shift
      if [[ "$1" =~ ^android26$|^qclinux$ ]]; then
        CPU_OS="$1"
      else
        echo "Error: Unsupported CPU OS '$1'"
        print_help
        exit 1
      fi
      ;;
    --help)
      print_help
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$1'"
      print_help
      exit 1
      ;;
  esac
  shift
done

# Validate compatibility
if [[ "$ARM_ARCH" == "armv9" && "$CPU_OS" == "qclinux" ]]; then
  echo "Error: qclinux is only supported with armv8 architecture."
  print_help
  exit 1
fi

if [ -z "$HEXAGON_SDK_ROOT" ]; then
    echo "Error: HEXAGON_SDK_ROOT is not set."
    exit 1
fi

# Extract algorithm name from parent directory
ALGO_NAME=$(basename "$(dirname "$(realpath "$0")")")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$CPU_OS" == "android26" ]; then
  # Set march flags based on ARM_ARCH
  if [ "$ARM_ARCH" == "armv8" ]; then
    MARCH_FLAGS="-march=armv8.2-a+dotprod+i8mm+fp16"
  elif [ "$ARM_ARCH" == "armv9" ]; then
    MARCH_FLAGS="-march=armv9.2-a+dotprod+i8mm+fp16+sme"
  fi

  # Check required environment variables
  if [ -z "$ANDROID_ROOT_DIR" ]; then
    echo "Error: ANDROID_ROOT_DIR is not set."
    exit 1
  fi

  CPU_CC=$ANDROID_ROOT_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang

  mkdir -p $SCRIPT_DIR/build/${ARM_ARCH}_android26

  $CPU_CC  -target aarch64-linux-android26 \
          $MARCH_FLAGS -ffast-math -O3 \
          -Wall -Wno-missing-braces  -I$SCRIPT_DIR/../../include  -I$HEXAGON_SDK_ROOT/incs \
          -fPIE -L$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64 \
          -L$ANDROID_ROOT_DIR/platforms/android-26/arch-arm64/usr/lib \
          -L$SCRIPT_DIR/../../lib/${ARM_ARCH}_android26 $SCRIPT_DIR/src/test_$ALGO_NAME.c \
          -llog -lm -lcdsprpc -fPIE $SCRIPT_DIR/../../lib/${ARM_ARCH}_android26/libsdkl.so \
          -o $SCRIPT_DIR/build/${ARM_ARCH}_android26/test_$ALGO_NAME
elif [ "$CPU_OS" == "qclinux" ]; then
  # Set march flags based on ARM_ARCH
  MARCH_FLAGS="-march=armv8.2-a+fp16  -DARM_ARCH_7A "

  # Check required environment variables
  if [ -z "$LV_TOOLS_DIR" ]; then
    echo "Error: LV_TOOLS_DIR is not set."
    exit 1
  fi

  CPU_CC=$LV_TOOLS_DIR/bin/aarch64-linux-gnu-gcc

  if ! command -v "$CPU_CC" >/dev/null 2>&1; then
     echo "Error: Compiler not found at $CPU_CC"
     echo "Please make sure LV_TOOLS_DIR is set correctly and linaro64 compiler is installed."
     exit 1
  fi   

  mkdir -p $SCRIPT_DIR/build/${ARM_ARCH}_qclinux

  $CPU_CC $MARCH_FLAGS  $SCRIPT_DIR/src/test_$ALGO_NAME.c $SCRIPT_DIR/../../lib/${ARM_ARCH}_qclinux/libsdkl.so \
           $HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/UbuntuARM_aarch64/libcdsprpc.so \
          -fPIC -Wall -Wno-missing-braces -DVERIFY_PRINT_ERROR -DUSE_SYSLOG -std=gnu99 -O2 -fno-strict-aliasing \
          -I$SCRIPT_DIR/../../include  -I$HEXAGON_SDK_ROOT/incs -isystem $LV_TOOLS_DIR/libc/usr/include  \
          -L$LV_TOOLS_DIR/lib/gcc/aarch64-linux-gnu/7.5.0   -L$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/UbuntuARM_aarch64  \
          -o $SCRIPT_DIR/build/${ARM_ARCH}_qclinux/test_$ALGO_NAME  -lm -lcdsprpc -lc -lstdc++ -lgcc_eh -lgcc
fi