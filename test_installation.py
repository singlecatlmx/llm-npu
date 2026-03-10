#!/usr/bin/env python3
"""
ExecuTorch 安装验证脚本
验证PyTorch、ExecuTorch、QNN和XNNPACK后端是否正确安装
"""

import sys
import torch
import torch.nn as nn


def test_pytorch():
    """测试PyTorch安装"""
    print(f"✅ PyTorch版本: {torch.__version__}")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
    return True


def test_executorch_core():
    """测试ExecuTorch核心功能"""
    try:
        import executorch
        print("✅ ExecuTorch导入成功")
        
        from executorch.exir import to_edge
        print("✅ EXIR模块导入成功")
        
        return True
    except Exception as e:
        print(f"❌ ExecuTorch核心测试失败: {e}")
        return False


def test_backends():
    """测试后端模块"""
    backends_status = {}
    
    # 测试QNN后端
    try:
        from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
        print("✅ QNN后端导入成功")
        backends_status['qnn'] = True
    except Exception as e:
        print(f"⚠️ QNN后端导入失败: {e}")
        backends_status['qnn'] = False
    
    # 测试XNNPACK后端
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        print("✅ XNNPACK后端导入成功")
        backends_status['xnnpack'] = True
    except Exception as e:
        print(f"❌ XNNPACK后端导入失败: {e}")
        backends_status['xnnpack'] = False
    
    return backends_status


def test_model_export():
    """测试模型导出功能"""
    try:
        from executorch.exir import to_edge
        
        # 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        example_input = (torch.randn(1, 5),)
        
        # 导出模型
        print("正在测试模型导出...")
        
        # 使用torch.export导出
        exported_program = torch.export.export(model, example_input)
        print("✅ torch.export导出成功")
        
        # 转换为Edge
        edge_program = to_edge(exported_program)
        print("✅ 转换为Edge程序成功")
        
        # 测试执行
        result = edge_program.module()(*example_input)
        print(f"✅ Edge程序执行成功，输出形状: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型导出测试失败: {e}")
        return False


def main():
    """主验证函数"""
    print("=" * 50)
    print("ExecuTorch 安装验证")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    # 1. PyTorch测试
    print("\n1. PyTorch测试:")
    if test_pytorch():
        success_count += 1
    
    # 2. ExecuTorch核心测试
    print("\n2. ExecuTorch核心测试:")
    if test_executorch_core():
        success_count += 1
    
    # 3. 后端测试
    print("\n3. 后端模块测试:")
    backends = test_backends()
    if any(backends.values()):
        success_count += 1
    
    # 4. 模型导出测试
    print("\n4. 模型导出测试:")
    if test_model_export():
        success_count += 1
    
    # 总结
    print("\n" + "=" * 50)
    print("验证结果:")
    print(f"成功: {success_count}/{total_tests} 项测试")
    
    if success_count == total_tests:
        print("🎉 ExecuTorch安装验证成功！")
        return 0
    elif success_count >= 3:
        print("⚠️ ExecuTorch基本安装成功，部分功能可能需要调试")
        return 0
    else:
        print("❌ ExecuTorch安装存在问题")
        return 1


if __name__ == "__main__":
    sys.exit(main())