#!/bin/bash
#===============================================================================
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#===============================================================================


print_help() {
  echo "Usage: $0 [--hex-arch <v73|v75|v79>] [--help]"
  echo ""
  echo "Options:"
  echo "  --hex-arch <v73|v75|v79>   Specify Hexagon architecture version (default: v73)"
  echo "  --help                     Show this help message"
}

# Default architecture
HEX_ARCH="v73"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hex-arch)
      shift
      if [[ "$1" =~ ^v73$|^v75$|^v79$ ]]; then
        HEX_ARCH="$1"
      else
        echo "Error: Unsupported architecture '$1'"
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

# Check HEXAGON_SDK_ROOT
if [ -z "$HEXAGON_SDK_ROOT" ]; then
  echo "Error: HEXAGON_SDK_ROOT is not set."
  exit 1
fi

if [ -z "$DEFAULT_HEXAGON_TOOLS_ROOT" ]; then
  echo "Error: DEFAULT_HEXAGON_TOOLS_ROOT is not set."
  exit 1
fi

if [ -z "$DEFAULT_TOOLS_VARIANT" ]; then
  echo "Error: DEFAULT_TOOLS_VARIANT is not set."
  exit 1
fi 

# Extract algorithm name from parent directory
ALGO_NAME=$(basename "$(dirname "$(realpath "$0")")")
TEST_FILE="test_${ALGO_NAME}.c"
OBJ_FILE="${TEST_FILE}.obj"
SO_NAME="lib${TEST_FILE%.*}_q.so"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NPU_CC=$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang

# Construct build directory name
BUILD_DIR="hexagon_${DEFAULT_TOOLS_VARIANT}_${HEX_ARCH}"
EXE_BUILD_DIR=$SCRIPT_DIR/$BUILD_DIR


mkdir -p "$EXE_BUILD_DIR"

# Compile
$NPU_CC -D${TEST_FILE%.*}_q_EXPORTS \
        -I$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include \
        -I$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/qurt \
        -I$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/posix \
        -I$HEXAGON_SDK_ROOT/ipc/fastrpc/rtld/ship/$BUILD_DIR \
        -I$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc \
        -I$SCRIPT_DIR/../../include \
        -I$HEXAGON_SDK_ROOT/rtos/qurt \
        -I$HEXAGON_SDK_ROOT/utils/examples \
        -isystem $HEXAGON_SDK_ROOT/incs \
        -isystem $HEXAGON_SDK_ROOT/incs/stddef \
        -isystem $HEXAGON_SDK_ROOT/ipc/fastrpc/incs \
        -m${HEX_ARCH} -G0 \
        -Wall -Werror -Wno-unused-function -fno-zero-initialized-in-bss -fdata-sections \
        -fpic -mllvm -enable-xqf-gen=true -mhvx -mhvx-length=128B -O3 \
        -fPIC -MD -MT $EXE_BUILD_DIR/$OBJ_FILE \
        -MF $EXE_BUILD_DIR/${OBJ_FILE}.d -o $EXE_BUILD_DIR/$OBJ_FILE -c $SCRIPT_DIR/src/$TEST_FILE

# Link
$NPU_CC -m${HEX_ARCH} -G0 -fpic -Wl,-Bsymbolic -Wl,-L$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/target/hexagon/lib/${HEX_ARCH}/G0/pic \
        -Wl,-L$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/target/hexagon/lib/ \
        -Wl,--no-threads -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -shared \
        -o $EXE_BUILD_DIR/$SO_NAME -Wl,-soname,$SO_NAME \
        -Wl,--start-group $EXE_BUILD_DIR/$OBJ_FILE \
         $SCRIPT_DIR/../../lib/$BUILD_DIR/libhexkl_macro.a -Wl,--end-group -lc
