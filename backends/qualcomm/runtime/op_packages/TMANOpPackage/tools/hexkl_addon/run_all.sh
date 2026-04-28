#!/bin/bash
#===============================================================================
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#===============================================================================


# Default values
HEX_ARCH="v73"
ARM_ARCH="armv8"
CPU_OS="android26"
LOG_FILE="run_all_log.txt"
REPORT_FILE="run_all_final_report.txt"

# Help message
show_help() {
    echo "Usage: $0 [--hex-arch <v73|v75|v79>] [--arm-arch <armv8|armv9>] [--cpu-os <android26|qclinux>] [--adb-flags <string>] [--help]"
    echo "  --hex-arch <v73|v75|v79>   Specify Hexagon architecture version (default: v73)"
    echo "  --arm-arch <armv8|armv9>   Set ARM architecture version (default: armv8)"
    echo "  --cpu-os <android26|qclinux> Set CPU OS (default: android26). qclinux supported only with armv8"
    echo "  --adb-flags <string>       Set ADB_FLAGS environment variable"
    echo "  --help                     Show this help message"
    echo ""
    echo "[RUN_ALL_SCRIPT] All stdout/stderr from examples will be saved to: $LOG_FILE"
    echo "[RUN_ALL_SCRIPT] Final report will be printed and saved to: $REPORT_FILE"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hex-arch)
            HEX_ARCH="$2"; shift 2;;
        --arm-arch)
            ARM_ARCH="$2"; shift 2;;
        --cpu-os)
            CPU_OS="$2"; shift 2;;
        --adb-flags)
            ADB_FLAGS="$2"; export ADB_FLAGS; shift 2;;
        --help)
            show_help; exit 0;;
        *)
            echo "[RUN_ALL_SCRIPT] Unknown option: $1"; show_help; exit 1;;
    esac
done

# Validate compatibility
if [[ "$ARM_ARCH" == "armv9" && "$CPU_OS" == "qclinux" ]]; then
    echo "[RUN_ALL_SCRIPT] Error: qclinux is only supported with armv8 architecture."
    exit 1
fi

if [ -z "$ADB_FLAGS" ]; then
    echo "[RUN_ALL_SCRIPT] Error: ADB_FLAGS environment variable is not set and --adb-flags was not provided."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/examples"
REPORT=""
echo "[RUN_ALL_SCRIPT] Starting run_all script..." > "$LOG_FILE"

for EXAMPLE_PATH in "$EXAMPLES_DIR"/*; do
    if [ -d "$EXAMPLE_PATH" ]; then
        EXAMPLE_NAME="$(basename "$EXAMPLE_PATH")"
        STATUS="Passed"
        echo "[RUN_ALL_SCRIPT] Executing example: $EXAMPLE_NAME with HEX_ARCH=$HEX_ARCH, ARM_ARCH=$ARM_ARCH, CPU_OS=$CPU_OS" | tee -a "$LOG_FILE"

        if [ -f "$EXAMPLE_PATH/run_simulator.sh" ]; then
            chmod +x "$EXAMPLE_PATH/run_simulator.sh"
            OUTPUT="$($EXAMPLE_PATH/run_simulator.sh --hex-arch $HEX_ARCH 2>&1)"
            echo "$OUTPUT" >> "$LOG_FILE"
            [ $? -ne 0 ] || echo "$OUTPUT" | grep -q "Test Failed" && STATUS="Failed"
        fi

        if [ -f "$EXAMPLE_PATH/run_android.sh" ]; then
            chmod +x "$EXAMPLE_PATH/run_android.sh"
            OUTPUT="$($EXAMPLE_PATH/run_android.sh --hex-arch $HEX_ARCH --arm-arch $ARM_ARCH --cpu-os $CPU_OS 2>&1)"
            echo "$OUTPUT" >> "$LOG_FILE"
            [ $? -ne 0 ] || echo "$OUTPUT" | grep -q "Test Failed" && STATUS="Failed"
        fi

        REPORT+="$EXAMPLE_NAME -> $STATUS\n"
    fi
done

echo -e "[RUN_ALL_SCRIPT] Final Report:\n$REPORT"
echo -e "$REPORT" > "$REPORT_FILE"
