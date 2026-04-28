#!/usr/bin/env bash
#
# run_device.sh -- Push and run ch05-hmx experiments on the device
#
# Prerequisites:
#   1. Phone connected via USB, adb available
#   2. build.sh has been run successfully
#   3. HexKL addon installed (see install_hexkl.sh)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXKL="$ROOT_DIR/tools/hexkl-addon"
DEVICE_DIR="/data/local/tmp/ch05"

# ---- Local file paths ----
PERFILL_LINEAR_TEST="$BUILD_DIR/libprefill_linear.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
HEXKL_SKEL="$HEXKL/lib/hexagon_toolv19_v75/libhexkl_skel.so"
TEST_DATA="$SCRIPT_DIR/test_data"

# ---- Check build artifacts ----
for f in "$PERFILL_LINEAR_TEST"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run build.sh first."
        exit 1
    fi
done

for f in "$RUN_MAIN" "$RUN_MAIN_SKEL" "$HEXKL_SKEL"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Check SDK / HexKL installation."
        exit 1
    fi
done

# ---- Check adb connection ----
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device connected. Check USB and run 'adb devices'."
    exit 1
fi

# ---- Push files to device ----
echo "=== Pushing files to device ==="
adb shell "mkdir -p $DEVICE_DIR"
adb push "$PERFILL_LINEAR_TEST"  "$DEVICE_DIR/"
adb push "$RUN_MAIN"             "$DEVICE_DIR/"
adb push "$RUN_MAIN_SKEL"        "$DEVICE_DIR/"
adb push "$HEXKL_SKEL"           "$DEVICE_DIR/"
adb push $TEST_DATA/*            "$DEVICE_DIR/" # 这一定要传输对的数据啊！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
adb shell "chmod +x $DEVICE_DIR/run_main_on_hexagon"

# ---- Enable FARF logging (required for FARF output to appear in logcat) ----
adb shell "echo '0x1f' > $DEVICE_DIR/run_main_on_hexagon.farf"

# ---- Experiment : TMAN Prefill Linear Test ----
echo ""
echo "=== Experiment : TMAN Prefill Linear Test ==="
echo "    Tests what minimum HMX init is needed for direct ASM compute"
echo ""
adb logcat -c
timeout 300 adb shell "cd $DEVICE_DIR && \
    DSP_LIBRARY_PATH=$DEVICE_DIR \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libprefill_linear.so" 2>&1 || echo "(timeout or failed)"

sleep 1
# adb logcat -d | grep "\[EXP\]" | sed 's/.*\[DU\]: //' | tail -40

adb logcat -s adsprpc -d  > log.txt
adb logcat -d | grep "\[DEBUG\]" > log.txt
