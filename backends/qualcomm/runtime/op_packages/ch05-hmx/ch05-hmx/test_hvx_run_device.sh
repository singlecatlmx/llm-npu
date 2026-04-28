#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXKL="$ROOT_DIR/tools/hexkl-addon"
DEVICE_DIR="/data/local/tmp/hvx_test"

TEST_SO="$BUILD_DIR/libtest_hvx_vtcm.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
HEXKL_SKEL="$HEXKL/lib/hexagon_toolv19_v75/libhexkl_skel.so"

adb shell "mkdir -p $DEVICE_DIR"
adb push "$TEST_SO"       "$DEVICE_DIR/"
adb push "$RUN_MAIN"      "$DEVICE_DIR/"
adb push "$RUN_MAIN_SKEL" "$DEVICE_DIR/"
adb push "$HEXKL_SKEL"    "$DEVICE_DIR/"

adb shell "chmod +x $DEVICE_DIR/run_main_on_hexagon"
adb shell "echo '0x1f' > $DEVICE_DIR/run_main_on_hexagon.farf"

echo "=== Running Tests ==="
adb logcat -c
adb shell "cd $DEVICE_DIR && \
    ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libtest_hvx_vtcm.so"

sleep 1
adb logcat -d | grep "\[PASS\]\|\[FAIL\]"