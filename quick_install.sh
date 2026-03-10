#!/bin/bash

# ExecuTorch快速安装脚本 - 已配置QNN SDK路径

echo "=== ExecuTorch 安装脚本 ==="
echo "QNN SDK 路径已永久配置到: /opt/qnn"

# 确保conda已初始化
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh

# 激活conda环境
conda activate executorch_env

# 设置QNN SDK路径（多重保险）
export QNN_SDK_ROOT="/opt/qnn"

# 显示当前配置
echo "当前环境: $CONDA_DEFAULT_ENV"
echo "QNN SDK: $QNN_SDK_ROOT"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"

# 运行安装
echo -e "\n=== 开始安装ExecuTorch ==="
CMAKE_ARGS="-DEXECUTORCH_BUILD_QNN=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_COREML=OFF -DEXECUTORCH_BUILD_MPS=OFF" ./install_executorch.sh

echo -e "\n=== 安装完成 ==="