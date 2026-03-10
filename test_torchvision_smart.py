#!/usr/bin/env python3
"""
torchvision 0.24.1 智能导入和使用脚本
尝试绕过ABI兼容性问题的多种方法
"""

import os
import sys
import warnings
import importlib

def disable_torchvision_ops():
    """尝试禁用有问题的torchvision算子"""
    try:
        # 方法1: 环境变量禁用
        os.environ['TORCHVISION_DISABLE_OPS'] = '1'
        
        # 方法2: 禁用特定的算子注册
        import torch
        if hasattr(torch, '_C') and hasattr(torch._C, '_dispatch_has_kernel_for_dispatch_key'):
            # 尝试monkey patch来绕过NMS检查
            original_has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
            
            def patched_has_kernel(op_name, dispatch_key):
                if 'torchvision::nms' in str(op_name):
                    return False  # 假装NMS算子不存在
                return original_has_kernel(op_name, dispatch_key)
            
            torch._C._dispatch_has_kernel_for_dispatch_key = patched_has_kernel
            print("✅ 已应用NMS算子绕过补丁")
            
        return True
    except Exception as e:
        print(f"⚠️ 补丁应用失败: {e}")
        return False

def smart_import_torchvision():
    """智能导入torchvision模块"""
    results = {}
    
    # 先应用补丁
    patch_applied = disable_torchvision_ops()
    
    # 禁用警告
    warnings.filterwarnings('ignore')
    
    print("=== torchvision 0.24.1 智能导入测试 ===")
    
    # 1. 尝试导入transforms（最常用且最可能成功）
    try:
        from torchvision import transforms
        # 测试基本功能
        t = transforms.ToTensor()
        resize = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 创建组合变换
        transform = transforms.Compose([resize, t, normalize])
        
        results['transforms'] = True
        print("✅ torchvision.transforms - 完全可用")
        
        # 测试更多transforms功能
        crop = transforms.CenterCrop(224)
        flip = transforms.RandomHorizontalFlip()
        results['transforms_advanced'] = True
        print("✅ torchvision.transforms高级功能 - 可用")
        
    except Exception as e:
        results['transforms'] = False
        print(f"❌ torchvision.transforms失败: {e}")
    
    # 2. 尝试导入utils模块
    try:
        from torchvision import utils
        results['utils'] = True
        print("✅ torchvision.utils - 可用")
    except Exception as e:
        results['utils'] = False
        print(f"❌ torchvision.utils失败: {e}")
    
    # 3. 尝试导入datasets模块
    try:
        from torchvision import datasets
        results['datasets'] = True
        print("✅ torchvision.datasets - 可用")
    except Exception as e:
        results['datasets'] = False
        print(f"❌ torchvision.datasets失败: {e}")
    
    # 4. 尝试分别导入模型模块
    model_modules = ['resnet', 'vgg', 'alexnet', 'densenet', 'inception', 'mobilenet']
    working_models = []
    
    for model_name in model_modules:
        try:
            module = importlib.import_module(f'torchvision.models.{model_name}')
            working_models.append(model_name)
            print(f"✅ torchvision.models.{model_name} - 可用")
        except Exception as e:
            print(f"❌ torchvision.models.{model_name}失败: {e}")
    
    results['models'] = working_models
    
    return results

def test_working_functionality(results):
    """测试实际可用的功能"""
    print(f"\n=== 功能测试 ===")
    
    if results.get('transforms', False):
        try:
            import torch
            from torchvision import transforms
            
            # 创建测试数据
            fake_image = torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)
            
            # 测试transforms链
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            result = transform(fake_image)
            print(f"✅ transforms功能测试成功: {result.shape}")
            
        except Exception as e:
            print(f"❌ transforms功能测试失败: {e}")
    
    if results.get('models') and len(results['models']) > 0:
        try:
            import torch
            # 尝试使用可用的模型
            for model_name in results['models'][:1]:  # 只测试第一个可用模型
                try:
                    module = importlib.import_module(f'torchvision.models.{model_name}')
                    if hasattr(module, model_name + '18'):
                        model_func = getattr(module, model_name + '18')
                    elif hasattr(module, model_name):
                        model_func = getattr(module, model_name)
                    else:
                        # 尝试获取模块中的第一个函数
                        funcs = [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]
                        if funcs:
                            model_func = getattr(module, funcs[0])
                        else:
                            continue
                    
                    model = model_func(pretrained=False)
                    x = torch.randn(1, 3, 224, 224)
                    
                    with torch.no_grad():
                        output = model(x)
                    
                    print(f"✅ {model_name}模型测试成功: {output.shape}")
                    break
                    
                except Exception as e:
                    print(f"⚠️ {model_name}模型测试失败: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ 模型功能测试失败: {e}")

def provide_usage_recommendations(results):
    """提供使用建议"""
    print(f"\n=== 使用建议 ===")
    
    working_count = sum(1 for k, v in results.items() if k != 'models' and v) + (1 if results.get('models') else 0)
    
    if working_count > 0:
        print(f"✅ torchvision 0.24.1 部分功能可用!")
        
        if results.get('transforms', False):
            print("📝 可以使用的功能:")
            print("   - torchvision.transforms: 数据预处理和增强")
            print("   - transforms.Compose: 组合多个变换")
            print("   - transforms.ToTensor, Normalize, Resize等")
        
        if results.get('models') and len(results['models']) > 0:
            print(f"   - 部分模型模块: {', '.join(results['models'])}")
        
        if results.get('datasets', False):
            print("   - torchvision.datasets: 标准数据集加载")
        
        print(f"\n💡 推荐的导入方式:")
        print("```python")
        if results.get('transforms', False):
            print("from torchvision import transforms")
        if results.get('models') and len(results['models']) > 0:
            for model in results['models'][:2]:
                print(f"from torchvision.models import {model}")
        print("```")
        
    else:
        print("❌ torchvision 0.24.1 完全不可用")
        print("💡 建议使用替代方案:")
        print("   - timm: pip install timm")
        print("   - PIL + OpenCV: pip install pillow opencv-python")

def main():
    """主测试函数"""
    print("=" * 60)
    print("torchvision 0.24.1 智能兼容性测试")
    print("=" * 60)
    
    # 智能导入测试
    results = smart_import_torchvision()
    
    # 功能测试
    test_working_functionality(results)
    
    # 提供建议
    provide_usage_recommendations(results)
    
    # 返回结果
    working_modules = sum(1 for k, v in results.items() if k != 'models' and v)
    if results.get('models'):
        working_modules += 1
    
    total_modules = 4  # transforms, utils, datasets, models
    success_rate = working_modules / total_modules
    
    print(f"\n=== 总结 ===")
    print(f"可用模块: {working_modules}/{total_modules} ({success_rate:.1%})")
    
    if success_rate >= 0.5:
        print("🎉 torchvision 0.24.1 基本可用!")
        return 0
    else:
        print("⚠️ torchvision 0.24.1 兼容性有限")
        return 1

if __name__ == "__main__":
    sys.exit(main())