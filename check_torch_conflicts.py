#!/usr/bin/env python3
"""
检查所有可能修改torch和torchvision版本的文件和脚本
"""

import os
import re
import glob
from pathlib import Path

def check_torch_version_references():
    """检查所有可能影响torch版本的文件"""
    
    base_dirs = [
        "/workspaces/tman/test/executorch",
        "/workspaces/tman/executorch", 
        "/workspaces/tman/executorch_cuda",
        "/workspaces/tman/executorch_python"
    ]
    
    dangerous_patterns = [
        r'torch\s*==\s*[0-9.]+',
        r'torch\s*>=\s*[0-9.]+', 
        r'torch\s*>\s*[0-9.]+',
        r'torchvision\s*==\s*[0-9.]+',
        r'torchvision\s*>=\s*[0-9.]+',
        r'torchvision\s*>\s*[0-9.]+',
        r'pip install.*torch(?:vision)?(?:\s*==\s*[0-9.]+)?',
        r'install_requires.*torch'
    ]
    
    file_patterns = [
        "**/*.py",
        "**/*.sh", 
        "**/*.txt",
        "**/*.toml",
        "**/*.yml",
        "**/*.yaml"
    ]
    
    results = {
        "high_risk": [],    # 会直接安装特定版本
        "medium_risk": [],  # 可能触发版本变更
        "low_risk": [],     # 仅引用但不安装
        "safe": []          # 已确认安全
    }
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        for pattern in file_patterns:
            files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
            
            for filepath in files:
                # 跳过某些目录
                if any(skip in filepath for skip in ['.git', '__pycache__', 'node_modules', '.pytest_cache']):
                    continue
                    
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # 检查是否包含危险模式
                    for pattern_str in dangerous_patterns:
                        matches = re.finditer(pattern_str, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            
                            risk_level = categorize_risk(filepath, match.group(), line_num, content)
                            
                            entry = {
                                'file': filepath,
                                'line': line_num,
                                'match': match.group(),
                                'pattern': pattern_str
                            }
                            
                            results[risk_level].append(entry)
                            
                except Exception as e:
                    print(f"读取文件失败: {filepath} - {e}")
    
    return results

def categorize_risk(filepath, match_text, line_num, content):
    """根据文件类型和内容分类风险等级"""
    
    # 高风险: 安装脚本中的版本指定
    if any(name in os.path.basename(filepath) for name in ['install', 'setup.py']) and '==' in match_text:
        # 但如果是我们修改过的文件且被注释掉，则是安全的
        lines = content.split('\n')
        if line_num <= len(lines):
            line_content = lines[line_num - 1].strip()
            if line_content.startswith('#'):
                return "safe"
        return "high_risk"
    
    # 中风险: requirements文件
    if 'requirements' in filepath and '==' in match_text:
        return "medium_risk"
        
    # 中风险: 配置文件中的版本指定
    if filepath.endswith(('.toml', '.yml', '.yaml')) and '==' in match_text:
        return "medium_risk"
    
    # 低风险: 代码中的引用
    if filepath.endswith('.py') and 'import' not in match_text and 'install' not in match_text:
        return "low_risk"
    
    # 默认中风险
    return "medium_risk"

def generate_report(results):
    """生成检查报告"""
    
    print("=" * 80)
    print("🔍 TORCH版本冲突风险检查报告")
    print("=" * 80)
    
    for risk_level in ['high_risk', 'medium_risk', 'low_risk', 'safe']:
        entries = results[risk_level]
        if not entries:
            continue
            
        risk_labels = {
            'high_risk': '🔴 高风险 (会直接覆盖torch版本)',
            'medium_risk': '🟡 中风险 (可能触发版本变更)', 
            'low_risk': '🟠 低风险 (仅引用不安装)',
            'safe': '✅ 安全 (已确认无害)'
        }
        
        print(f"\n{risk_labels[risk_level]}")
        print("-" * 60)
        
        for entry in entries:
            rel_path = entry['file'].replace('/workspaces/tman/', '')
            print(f"📁 {rel_path}:{entry['line']}")
            print(f"   匹配: {entry['match']}")
            print()
    
    # 总结
    print("=" * 80)
    print("📊 风险统计")
    print("=" * 80)
    for risk_level, entries in results.items():
        risk_names = {
            'high_risk': '高风险',
            'medium_risk': '中风险',
            'low_risk': '低风险', 
            'safe': '安全'
        }
        print(f"{risk_names[risk_level]}: {len(entries)} 项")

def main():
    """主检查函数"""
    print("开始检查torch版本冲突风险...")
    
    results = check_torch_version_references()
    generate_report(results)
    
    # 给出建议
    high_risk_count = len(results['high_risk'])
    medium_risk_count = len(results['medium_risk'])
    
    print("\n" + "=" * 80)
    print("💡 建议措施")
    print("=" * 80)
    
    if high_risk_count > 0:
        print("🚨 发现高风险项目，需要立即处理:")
        print("   1. 检查所有高风险文件")
        print("   2. 注释掉或修改版本指定") 
        print("   3. 使用--no-deps参数安装")
    
    if medium_risk_count > 0:
        print("⚠️ 中风险项目建议检查:")
        print("   1. 备份当前torch版本")
        print("   2. 运行前检查pip freeze输出")
        print("   3. 必要时使用版本锁定")
    
    print("\n🔒 版本保护措施:")
    print("   1. 使用 pip freeze > current_versions.txt 备份")
    print("   2. 在关键脚本中添加版本检查") 
    print("   3. 使用虚拟环境隔离")

if __name__ == "__main__":
    main()