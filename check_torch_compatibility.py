#!/usr/bin/env python3
"""
PyTorch生态系统兼容性检查和修复脚本
"""

import sys
import subprocess
import importlib.util

def check_module_import(module_name, verbose=True):
    """检查模块是否可以正常导入"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            if verbose:
                print(f"❌ {module_name}: 模块未安装")
            return False, "未安装"
        
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Unknown')
        if verbose:
            print(f"✅ {module_name}: {version}")
        return True, version
    
    except ImportError as e:
        if verbose:
            print(f"❌ {module_name}: 导入失败 - {str(e)}")
        return False, str(e)
    except Exception as e:
        if verbose:
            print(f"⚠️ {module_name}: 其他错误 - {str(e)}")
        return False, str(e)

def test_torch_functionality():
    """测试PyTorch基本功能"""
    try:
        import torch
        print(f"\n=== PyTorch功能测试 ===")
        
        # 基本tensor操作
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.matmul(x, y)
        print(f"✅ 基本tensor运算: {z.shape}")
        
        # 模型定义
        import torch.nn as nn
        model = nn.Linear(10, 1)
        input_data = torch.randn(1, 10)
        output = model(input_data)
        print(f"✅ 神经网络模块: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch功能测试失败: {e}")
        return False

def test_torchvision_functionality():
    """测试torchvision功能"""
    try:
        import torchvision
        import torchvision.transforms as transforms
        print(f"\n=== torchvision功能测试 ===")
        
        # 基本transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        print("✅ transforms定义成功")
        
        return True
    except Exception as e:
        print(f"❌ torchvision功能测试失败: {e}")
        return False

def suggest_fixes():
    """建议修复方案"""
    print(f"\n=== 修复建议 ===")
    print("1. 如果只使用ExecuTorch核心功能，可以忽略torchvision/torchaudio错误")
    print("2. 卸载有问题的组件: pip uninstall torchvision torchaudio")
    print("3. 重新安装CPU版本: pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("4. 或者使用--no-deps安装: pip install torchvision torchaudio --no-deps")

def main():
    """主检查函数"""
    print("=" * 60)
    print("PyTorch生态系统兼容性检查")
    print("=" * 60)
    
    modules_to_check = [
        'torch',
        'torchvision', 
        'torchaudio',
        'executorch',
        'numpy',
        'PIL'
    ]
    
    success_count = 0
    results = {}
    
    print("=== 模块导入检查 ===")
    for module in modules_to_check:
        success, info = check_module_import(module)
        results[module] = (success, info)
        if success:
            success_count += 1
    
    # 功能测试
    torch_works = test_torch_functionality()
    torchvision_works = test_torchvision_functionality()
    
    print(f"\n=== 总结 ===")
    print(f"模块导入成功: {success_count}/{len(modules_to_check)}")
    print(f"PyTorch功能: {'✅' if torch_works else '❌'}")
    print(f"torchvision功能: {'✅' if torchvision_works else '❌'}")
    
    # 核心ExecuTorch是否可用
    executorch_ok = results.get('executorch', (False, ''))[0] and torch_works
    print(f"ExecuTorch可用性: {'✅' if executorch_ok else '❌'}")
    
    if not torch_works or not executorch_ok:
        suggest_fixes()
        return 1
    
    if not torchvision_works:
        print("\n⚠️ torchvision有问题，但ExecuTorch核心功能可用")
        suggest_fixes()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())