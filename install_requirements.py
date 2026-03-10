# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import platform
import re
import subprocess
import sys
import tempfile
import shutil


def python_is_compatible():
    # Scrape the version range from pyproject.toml, which should be in the current directory.
    version_specifier = None
    with open("pyproject.toml", "r") as file:
        for line in file:
            if line.startswith("requires-python"):
                match = re.search(r'"([^"]*)"', line)
                if match:
                    version_specifier = match.group(1)
                    break

    if not version_specifier:
        print(
            "WARNING: Skipping python version check: version range not found",
            file=sys.stderr,
        )
        return False

    # Install the packaging module if necessary.
    try:
        import packaging
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "packaging"], check=True
        )
    # Compare the current python version to the range in version_specifier. Exits
    # with status 1 if the version is not compatible, or with status 0 if the
    # version is compatible or the logic itself fails.
    try:
        import packaging.specifiers
        import packaging.version

        python_version = packaging.version.parse(platform.python_version())
        version_range = packaging.specifiers.SpecifierSet(version_specifier)
        if python_version not in version_range:
            print(
                f'ERROR: ExecuTorch does not support python version {python_version}: must satisfy "{version_specifier}"',
                file=sys.stderr,
            )
            return False
    except Exception as e:
        print(f"WARNING: Skipping python version check: {e}", file=sys.stderr)
    return True


# The pip repository that hosts nightly torch packages.
TORCH_NIGHTLY_URL = "https://download.pytorch.org/whl/nightly/cpu"


# Since ExecuTorch often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
#
# NOTE: If you're changing, make the corresponding change in .ci/docker/ci_commit_pins/pytorch.txt
# by picking the hash from the same date in https://hud.pytorch.org/hud/pytorch/pytorch/nightly/
NIGHTLY_VERSION = "dev20250325"


def install_pytorch_from_source(target_date="2025-03-25"):
    """Install PyTorch from source for the specific date."""
    print(f"Installing PyTorch from source for date: {target_date}")
    
    # 使用持久化目录，不要每次都删除
    pytorch_dir = "/tmp/pytorch_build_cache"
    
    # 如果目录已存在且是git仓库，就更新而不是重新克隆
    if os.path.exists(pytorch_dir) and os.path.exists(os.path.join(pytorch_dir, ".git")):
        print("找到现有PyTorch源码缓存，更新中...")
        original_cwd = os.getcwd()
        os.chdir(pytorch_dir)
        try:
            subprocess.run(["git", "fetch", "origin"], check=True)
            subprocess.run(["git", "reset", "--hard", "HEAD"], check=True)
        finally:
            os.chdir(original_cwd)
    else:
        # 删除可能损坏的目录并重新克隆
        if os.path.exists(pytorch_dir):
            shutil.rmtree(pytorch_dir)
        
        print("克隆PyTorch仓库...")
        subprocess.run([
            "git", "clone", "--recursive", 
            "https://github.com/pytorch/pytorch.git", 
            pytorch_dir
        ], check=True)
    
    # Find commit for the target date
    original_cwd = os.getcwd()
    os.chdir(pytorch_dir)
    
    try:
        # Get commit for the target date
        result = subprocess.run([
            "git", "rev-list", "-n", "1", 
            f"--before='{target_date} 23:59:59'", "main"
        ], capture_output=True, text=True, check=True)
        
        commit_hash = result.stdout.strip()
        print(f"Found commit for {target_date}: {commit_hash}")
        
        # Checkout the specific commit
        subprocess.run(["git", "checkout", commit_hash], check=True)
        subprocess.run(["git", "submodule", "sync"], check=True)
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)
        
        # Build and install
        env = os.environ.copy()
        env['USE_CUDA'] = '0'  # CPU only build
        env['BUILD_TEST'] = '0'
        env['USE_MKLDNN'] = '0'
        env['USE_NNPACK'] = '0'
        
        print("Building PyTorch from source (this may take a while)...")
        subprocess.run([sys.executable, "setup.py", "install"], env=env, check=True)
        
        print("PyTorch installed successfully from source!")
        
    finally:
        os.chdir(original_cwd)
        # 不删除源码缓存，保留以备后用
        print(f"PyTorch源码已保存到: {pytorch_dir}")


def check_torch_version():
    """检查当前安装的PyTorch版本"""
    try:
        import torch
        version = torch.__version__
        print(f"当前PyTorch版本: {version}")
        
        # 检查是否是我们期望的源码版本
        if "git" in version and "2.8.0" in version:
            print("✅ 检测到正确的源码构建版本")
            return True
        else:
            print("❌ 当前版本不是期望的源码版本")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def install_requirements(use_pytorch_nightly):
    # 首先检查是否已经安装了正确的PyTorch版本
    print("=== 检查当前PyTorch版本 ===")
    if check_torch_version():
        print("✅ 已有正确的PyTorch版本，跳过重新安装")
        torch_source_installed = True
    else:
        # 强制只从源码安装PyTorch 2025-03-25版本
        print("=== 强制从Git源码构建PyTorch ===")
        print("跳过PyPI安装，直接构建2025-03-25版本的PyTorch...")
        
        try:
            install_pytorch_from_source("2025-03-25")
            print("✅ PyTorch源码构建完成!")
            torch_source_installed = True
        except Exception as e:
            print(f"❌ 源码构建失败: {e}")
            print("ERROR: 无法继续，必须使用2025-03-25版本的PyTorch")
            sys.exit(1)
    
    # 永远不通过pip安装torch - 只使用我们从源码构建的版本
    EXIR_REQUIREMENTS = []
    
    # 对于其他torch相关包，使用兼容的stable版本，避免版本冲突
    VISION_REQUIREMENTS = ["torchvision"]  # 使用stable版本
    EXAMPLES_REQUIREMENTS = ["torchaudio"]  # 使用stable版本

    # 首先安装基础requirements (不包含依赖torch的包)
    print("=== 安装基础依赖包 ===")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements-dev.txt",  # 只安装开发依赖，不包含torch相关
        ],
        check=True,
    )
    
    # 然后安装torch相关的辅助包
    print("=== 安装torch相关辅助包 ===")
    torch_related_packages = VISION_REQUIREMENTS + EXAMPLES_REQUIREMENTS
    if torch_related_packages:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip", 
                "install",
                *torch_related_packages,
                "--no-deps",  # 关键：不安装依赖，避免重新安装torch
            ],
            check=True,
        )
    
    # 最后单独安装需要torch的examples包
    print("=== 安装需要torch的示例包 ===")
    examples_packages = ["timm==1.0.7", "torchsr==1.0.4", "transformers==4.47.1"]
    subprocess.run(
        [
            sys.executable,
            "-m", 
            "pip",
            "install",
            *examples_packages,
            "--no-deps",  # 不安装依赖，使用我们已有的torch
        ],
        check=True,
    )

    LOCAL_REQUIREMENTS = [
        "third-party/ao",  # We need the latest kernels for fast iteration, so not relying on pypi.
        "extension/llm/tokenizers",  # TODO(larryliu0820): Setup a pypi package for this.
    ]

    # Install packages directly from local copy instead of pypi.
    # This is usually not recommended.
    new_env = os.environ.copy()
    new_env["USE_CPP"] = "1"  # install torchao kernels
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            # Without --no-build-isolation, setup.py can't find the torch module.
            "--no-build-isolation",
            *LOCAL_REQUIREMENTS,
        ],
        env=new_env,
        check=True,
    )


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-pt-pinned-commit",
        action="store_true",
        help="build from the pinned PyTorch commit instead of nightly",
    )
    args = parser.parse_args(args)
    install_requirements(use_pytorch_nightly=not bool(args.use_pt_pinned_commit))


if __name__ == "__main__":
    # Before doing anything, cd to the directory containing this script.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not python_is_compatible():
        sys.exit(1)
    main(sys.argv[1:])
