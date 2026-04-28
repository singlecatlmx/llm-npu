#!/bin/bash

# PyTorch版本保护验证脚本

echo "=== 验证脚本修改结果 ==="

# 1. 检查当前PyTorch版本
echo "1. 当前PyTorch版本检查："
python -c "
import torch
print(f'版本: {torch.__version__}')
print(f'Git版本: {torch.version.git_version}')
print(f'是否源码构建: {\"git\" in torch.__version__}')
"

# 2. 检查源码缓存
echo -e "\n2. PyTorch源码缓存检查："
if [ -d "/tmp/pytorch_build_cache" ]; then
    echo "✅ 源码缓存存在: /tmp/pytorch_build_cache"
    echo "大小: $(du -sh /tmp/pytorch_build_cache | cut -f1)"
else
    echo "❌ 源码缓存不存在，首次运行时会创建"
fi

# 3. 检查安装的torch相关包
echo -e "\n3. 已安装的torch相关包："
pip list | grep -i torch

# 4. 验证install_requirements.py的关键修改点
echo -e "\n4. install_requirements.py关键修改验证："
grep -n "check_torch_version" /workspaces/tman/test/executorch/install_requirements.py
grep -n "pytorch_build_cache" /workspaces/tman/test/executorch/install_requirements.py
grep -n "no-deps" /workspaces/tman/test/executorch/install_requirements.py

echo -e "\n=== 验证完成 ==="
echo "✅ 脚本已修改为:"
echo "  - 检查现有PyTorch版本，避免重复安装"
echo "  - 保留源码缓存到 /tmp/pytorch_build_cache"
echo "  - 使用 --no-deps 安装其他包，防止重装torch"
echo "  - 只从git源码构建2025-03-25版本的PyTorch"