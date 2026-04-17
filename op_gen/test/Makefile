# TMANAdd Test - 双平台构建 + 离线编译工作流
#
# 三种工作模式:
#   1. x86 模拟器 (一体化):        make run_x86
#   2. 离线编译 (生成 context bin): make compile_context
#   3. 设备执行 (加载 context bin): make run_android
#
# 完整设备测试工作流:
#   make compile_context    # x86 上离线编译, 生成 tman_ctx.bin
#   make run_android        # 推送到设备 + 执行
#
# 前置要求:
#   export QNN_SDK_ROOT=/home/tang/QNN/qairt/2.40.0.251030
#   export ANDROID_NDK_ROOT=/home/tang/android-ndk-r28b  (仅 android 目标需要)

# ============================================================
# 路径配置
# ============================================================
QNN_SDK_ROOT     ?= /home/tang/QNN/qairt/2.40.0.251030
ANDROID_NDK_ROOT ?= /home/tang/android-ndk-r28b

QNN_INCLUDE      := $(QNN_SDK_ROOT)/include/QNN

# 库路径
QNN_LIB_ANDROID  := $(QNN_SDK_ROOT)/lib/aarch64-android
QNN_LIB_HEX_V75  := $(QNN_SDK_ROOT)/lib/hexagon-v75/unsigned
QNN_LIB_X86      := $(QNN_SDK_ROOT)/lib/x86_64-linux-clang

# Op Package 路径
OP_PKG_DIR       := ../TMANOpPackage
OP_PKG_HEX_V75   := $(OP_PKG_DIR)/build/hexagon-v75/libQnnTMANOpPackage.so
OP_PKG_ARM64     := $(OP_PKG_DIR)/build/aarch64-android/libQnnTMANOpPackage.so
OP_PKG_X86       := $(OP_PKG_DIR)/build/x86_64-linux-clang/libQnnTMANOpPackage.so

# 设备工作目录
DEVICE_DIR       := /data/local/tmp/tman_test

# Context binary 和 metadata 文件名
CTX_BIN          := tman_ctx.bin
CTX_META         := tman_ctx.meta

# ============================================================
# 编译器配置
# ============================================================
NDK_TOOLCHAIN   := $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64
AARCH64_CXX     := $(NDK_TOOLCHAIN)/bin/clang++ \
                    --target=aarch64-none-linux-android28 \
                    --sysroot=$(NDK_TOOLCHAIN)/sysroot

X86_CXX         := clang++

COMMON_CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
                   -I$(QNN_INCLUDE)

ANDROID_LDFLAGS := -ldl -llog
X86_LDFLAGS     := -ldl -Wl,-rpath,'$$ORIGIN'

# ============================================================
# 目标文件
# ============================================================
TARGET        := test_tman_add
SRC           := test_tman_add.cpp
BUILD_ANDROID := build_android
BUILD_X86     := build_x86

# ============================================================
# 默认目标
# ============================================================
.PHONY: all android x86 push run_android run_x86 clean help \
        compile_context check-env-common check-env-android check-env-x86

all: android x86
	@echo ""
	@echo "[OK] Both targets built."

# ============================================================
# 编译目标
# ============================================================
android: check-env-common check-env-android $(BUILD_ANDROID)/$(TARGET)
	@echo "Built: $(BUILD_ANDROID)/$(TARGET)  (aarch64-android)"

$(BUILD_ANDROID)/$(TARGET): $(SRC) | $(BUILD_ANDROID)
	$(AARCH64_CXX) $(COMMON_CXXFLAGS) -o $@ $< $(ANDROID_LDFLAGS)

$(BUILD_ANDROID):
	mkdir -p $(BUILD_ANDROID)

x86: check-env-common check-env-x86 $(BUILD_X86)/$(TARGET)
	@echo "Built: $(BUILD_X86)/$(TARGET)  (x86_64-linux)"

$(BUILD_X86)/$(TARGET): $(SRC) | $(BUILD_X86)
	$(X86_CXX) $(COMMON_CXXFLAGS) -o $@ $< $(X86_LDFLAGS)

$(BUILD_X86):
	mkdir -p $(BUILD_X86)

# ============================================================
# x86: 模拟器一体化 (编译+执行, 无需 context binary)
# ============================================================
run_x86: x86
	@echo ""
	@echo "========================================"
	@echo "  Running on PC (QNN HTP simulator)..."
	@echo "========================================"
	@test -f "$(OP_PKG_X86)" || \
	    (echo "[ERROR] $(OP_PKG_X86) not found. Run: cd ../TMANOpPackage && make htp_x86" && exit 1)
	LD_LIBRARY_PATH="$(QNN_LIB_X86):$$LD_LIBRARY_PATH" \
	    ./$(BUILD_X86)/$(TARGET) \
	    $(QNN_LIB_X86)/libQnnHtp.so \
	    $(OP_PKG_X86)

# ============================================================
# 离线编译: 在 x86 host 上生成 context binary
# ============================================================
compile_context: x86
	@echo ""
	@echo "========================================"
	@echo "  Compiling context binary on host..."
	@echo "========================================"
	@test -f "$(OP_PKG_X86)" || \
	    (echo "[ERROR] $(OP_PKG_X86) not found. Run: cd ../TMANOpPackage && make htp_x86" && exit 1)
	LD_LIBRARY_PATH="$(QNN_LIB_X86):$$LD_LIBRARY_PATH" \
	    ./$(BUILD_X86)/$(TARGET) --compile \
	    $(QNN_LIB_X86)/libQnnHtp.so \
	    $(OP_PKG_X86) \
	    $(CTX_BIN)

# ============================================================
# Android: Push 到设备
# ============================================================
push: android
	@echo "========================================"
	@echo "  Pushing files to device: $(DEVICE_DIR)"
	@echo "========================================"
	@test -f "$(CTX_BIN)" || \
	    (echo "[ERROR] $(CTX_BIN) not found. Run: make compile_context" && exit 1)
	@test -f "$(CTX_META)" || \
	    (echo "[ERROR] $(CTX_META) not found. Run: make compile_context" && exit 1)
	adb shell "mkdir -p $(DEVICE_DIR)/hexagon"
	adb push $(BUILD_ANDROID)/$(TARGET) $(DEVICE_DIR)/
	adb push $(CTX_BIN) $(DEVICE_DIR)/
	adb push $(CTX_META) $(DEVICE_DIR)/
	adb push $(QNN_LIB_ANDROID)/libQnnHtp.so $(DEVICE_DIR)/
	adb push $(QNN_LIB_ANDROID)/libQnnHtpPrepare.so $(DEVICE_DIR)/
	adb push $(QNN_LIB_ANDROID)/libQnnHtpV75Stub.so $(DEVICE_DIR)/ 2>/dev/null || true
	@# HTP skel 库 (DSP 侧)
	@for f in $(QNN_LIB_HEX_V75)/libQnnHtpV75Skel.so \
	          $(QNN_LIB_HEX_V75)/libqnnhtpv75.cat; do \
	    if [ -f "$$f" ]; then adb push "$$f" $(DEVICE_DIR)/; fi; \
	done
	@# Op package: hexagon v75 (DSP 执行内核)
	@if [ -f "$(OP_PKG_HEX_V75)" ]; then \
	    adb push $(OP_PKG_HEX_V75) $(DEVICE_DIR)/hexagon/libQnnTMANOpPackage.so; \
	    echo "[OK] Hexagon op package pushed (DSP side)"; \
	else \
	    echo "[WARN] $(OP_PKG_HEX_V75) not found. Run: cd ../TMANOpPackage && make htp_v75"; \
	fi
	adb shell "chmod +x $(DEVICE_DIR)/$(TARGET)"
	@echo "[OK] All files pushed to $(DEVICE_DIR)"

# ============================================================
# Android: 在设备上执行 (从 context binary 恢复)
# ============================================================
run_android: push
	@echo ""
	@echo "========================================"
	@echo "  Running test on device..."
	@echo "========================================"
	adb shell "cd $(DEVICE_DIR) && \
    export LD_LIBRARY_PATH=$(DEVICE_DIR) && \
    export ADSP_LIBRARY_PATH=\"$(DEVICE_DIR)/hexagon;$(DEVICE_DIR);/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp\" && \
    ./$(TARGET) --execute \
        ./libQnnHtp.so \
        ./hexagon/libQnnTMANOpPackage.so \
        ./$(CTX_BIN)"

# ============================================================
# 清理
# ============================================================
clean:
	rm -rf $(BUILD_ANDROID) $(BUILD_X86) $(CTX_BIN) $(CTX_META)
	@echo "Cleaned all build artifacts"

# ============================================================
# 环境检查
# ============================================================
check-env-common:
	@test -d "$(QNN_SDK_ROOT)" || \
	    (echo "[ERROR] QNN_SDK_ROOT not found: $(QNN_SDK_ROOT)" && exit 1)
	@test -f "$(QNN_INCLUDE)/QnnInterface.h" || \
	    (echo "[ERROR] QNN headers not found at $(QNN_INCLUDE)" && exit 1)

check-env-android:
	@test -d "$(ANDROID_NDK_ROOT)" || \
	    (echo "[ERROR] ANDROID_NDK_ROOT not found: $(ANDROID_NDK_ROOT)" && exit 1)

check-env-x86:
	@command -v $(firstword $(X86_CXX)) >/dev/null 2>&1 || \
	    (echo "[ERROR] clang++ not found" && exit 1)

# ============================================================
# 帮助
# ============================================================
help:
	@echo ""
	@echo "TMANAdd Test - 使用说明"
	@echo ""
	@echo "  x86 模拟器测试 (最简单, 无需设备):"
	@echo "    make run_x86"
	@echo ""
	@echo "  设备端测试 (完整工作流):"
	@echo "    make compile_context   # Step 1: x86 上生成 context binary"
	@echo "    make run_android       # Step 2: 推送到设备并执行"
	@echo ""
	@echo "  其他命令:"
	@echo "    make all               # 编译两个平台"
	@echo "    make clean             # 清理"
	@echo ""
	@echo "  前置: 编译 op package"
	@echo "    cd ../TMANOpPackage"
	@echo "    make htp_x86           # x86 模拟器和离线编译用"
	@echo "    make htp_v75           # 设备 DSP 执行用"
	@echo ""
