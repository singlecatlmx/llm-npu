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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_NAME=$(basename "$SCRIPT_DIR")
SO_NAME="libtest_${ALGO_NAME}_q.so"

# Construct build directory name
BUILD_DIR="$SCRIPT_DIR/hexagon_${DEFAULT_TOOLS_VARIANT}_${HEX_ARCH}"

# Generate config files
echo "$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" > "$BUILD_DIR/q6ss.cfg"
echo "$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/lib/iss/l2vic.so 32 0xab010000" >> "$BUILD_DIR/q6ss.cfg"
echo "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/debugger/lnx64/qurt_model.so" > "$BUILD_DIR/osam.cfg"

# Run simulation
$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-sim \
  -m${HEX_ARCH}na_1 --simulated_returnval --usefs "$BUILD_DIR" \
  --pmu_statsfile "$BUILD_DIR/pmu_stats.txt" --cosim_file "$BUILD_DIR/q6ss.cfg" \
  --l2tcm_base 0xd800 --rtos "$BUILD_DIR/osam.cfg" \
  "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/sdksim_bin/runelf.pbn" \
  -- "$HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship/hexagon_${DEFAULT_TOOLS_VARIANT}_${HEX_ARCH}/run_main_on_hexagon_sim" \
  --"$BUILD_DIR/$SO_NAME" 100
