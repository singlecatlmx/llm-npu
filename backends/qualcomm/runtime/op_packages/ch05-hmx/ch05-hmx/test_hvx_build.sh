#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXKL="$ROOT_DIR/tools/hexkl-addon"
HEXAGON_TOOLS="$HEXAGON_SDK/tools/HEXAGON_Tools/19.0.04/Tools"
HEX_CC="$HEXAGON_TOOLS/bin/hexagon-clang"

mkdir -p "$BUILD_DIR"

HEX_CFLAGS="-mv79 -G0 -fpic -mhvx -mhvx-length=128B -O2 -mhmx \
    -Wall -fno-zero-initialized-in-bss"

HEX_INCLUDES="\
    -I$HEXAGON_SDK/rtos/qurt/computev75/include \
    -I$HEXAGON_SDK/incs \
    -I$HEXAGON_SDK/incs/stddef \
    -I$HEXKL/include" 

HEX_LDFLAGS="-shared -Wl,-Bsymbolic -Wl,--no-threads"

echo "=== Compiling HVX VTCM Test"
$HEX_CC $HEX_CFLAGS $HEX_INCLUDES \
    -c "$SCRIPT_DIR/src/test_hvx_vtcm.c" \
    -o "$BUILD_DIR/test_hvx_vtcm.obj"

echo "=== Linking libtest_hvx_vtcm.so"
$HEX_CC -mv79 $HEX_LDFLAGS \
    "$BUILD_DIR/test_hvx_vtcm.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" -Wl,--end-group \
    -o "$BUILD_DIR/libtest_hvx_vtcm.so"