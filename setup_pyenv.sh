#!/bin/bash

setup_env_and_install() {
    source .venv/bin/activate || { echo "❌ 激活虚拟环境失败！"; }
    [[ -d ".venv" ]] && { echo "✅ 虚拟环境已存在，跳过安装步骤..."; return 0; }
    
    echo "🔧 开始环境安装和依赖配置..."
    set -euo pipefail
    
    echo "📦 1. 开始设置环境依赖安装..."
    
    echo "🏗️  创建新的虚拟环境..."
    uv venv -p '/usr/bin/python' --system-site-packages || { echo "❌ 创建虚拟环境失败！"; return 1; }
    source .venv/bin/activate || { echo "❌ 激活虚拟环境失败！"; return 1; }
    uv pip install pybind11   
    uv pip install -r  /root/requirements.txt
    
    echo "✅ 2. 虚拟环境已激活"
    local projects=(
        "/root/dependency_data_tool_source_code" 
        "/root/bucket_vast" 
        "my_utils"
        "."
    )
    echo "📋 3. 开始安装项目依赖..."
    for project in "${projects[@]}"; do
        if [[ -d "$project" ]]; then
            echo "📦 正在安装项目: $project"
            uv pip install -e "$project" || { echo "❌ 安装项目 $project 失败！"; return 1; }
        else
            echo "⚠️  项目目录不存在，跳过: $project"
        fi
    done
    
   
    uv pip install omegaconf
    uv pip install yunchang-0.6.0-py3-none-any.whl
    uv pip install einops
    uv pip install --no-build-isolation transformer_engine[pytorch]
    
    uv pip install tensordict
    uv pip install etcd3
    uv pip install joblib
    uv pip install xgboost
    uv pip install scikit-learn


    cp -r megatron/core/models/vae/autoencoder_kl* .venv/lib/python3.10/site-packages/diffusers/models/autoencoders/
    cp -r megatron/core/models/vae/tile_parallel_utils.py .venv/lib/python3.10/site-packages/diffusers/models/autoencoders/
    echo "Configuring Git global user..."
    echo "Git user configured."
    echo "----------------------------------------"    

    echo "🎉 所有依赖安装成功！"
    echo "📁 虚拟环境位置: $(pwd)/.venv"
}


setup_env_and_install
