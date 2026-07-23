#!/bin/bash

# 解压 VSCode Server
tar -zxf /workspace/tools/vscode-server-linux-x64.tar.gz -C .
mkdir -p ~/.vscode-server/bin
cp -r vscode-server-linux-x64 ~/.vscode-server/bin/8b3775030ed1a69b13e4f4c628c612102e30a681

# alias code="code-server"
export PATH=$PATH:~/.vscode-server/bin/8b3775030ed1a69b13e4f4c628c612102e30a681/bin
export USER=anon

echo $PATH
# 目标目录
VSCODE_SERVER_DIR="$HOME/.vscode-server/extensions/"

# 确保目标目录存在
mkdir -p "$VSCODE_SERVER_DIR"

# .vsix 文件路径（在容器内的路径）
VSIX_SOURCE_DIR="/workspace/vsixs/extensions/"  # 替换为容器内实际的 .vsix 文件所在路径


cp -r "$VSIX_SOURCE_DIR" "$VSCODE_SERVER_DIR"

# # 检查是否有 .vsix 
# if [ -d "$VSIX_SOURCE_DIR" ]; then
#     echo "拷贝 .vsix 文件到 $VSCODE_SERVER_DIR ..."
#     cp "$VSIX_SOURCE_DIR"/*.vsix "$VSCODE_SERVER_DIR/"
# else
#     echo "未找到 .vsix 文件目录: $VSIX_SOURCE_DIR"
#     exit 1
# fi

# # 安装所有拷贝的 .vsix 文件
# for VSIX_FILE in "$VSCODE_SERVER_DIR"/*.vsix; do
#     if [ -f "$VSIX_FILE" ]; then
#         echo "安装扩展: $VSIX_FILE"
#         code-server --install-extension "$VSIX_FILE" --force
#     else
#         echo "未找到 .vsix 文件，跳过安装。"
#     fi
# done

# # 检查是否安装成功
# echo "已安装的扩展:"
# code-server --list-extensions --show-versions
# export NCCL_IB_HCA=$(/root.sh) #必须有，不修改，本次升级改动
# echo NCCL_IB_HCA=$NCCL_IB_HCA
# cp -r <path>/teleai_data_tool_source_code /workspace/teleai_data_tool
# pip install -e /workspace/teleai_data_tool
# cd <repo>/vast3/vast
# pip install -e .
# export NCCL_NVLS_ENABLE=0&&export NCCL_CROSS_NIC=0&&export NCCL_DEBUG=INFO&&export NCCL_ALGO=RING&&python tools/train.py projects/hunyuanvideo/configs/hunyuanvideo_i2vhy.py


bash run.sh

# cd <repo>/vast2/vast

# 安装本项目
# pip install -v -e .



# 安装数据工具依赖, $USER是个人研发云的账号,即邮箱前缀
# pip install ../teleai_data_tool

echo 'install completed!'






# bash run.sh

# 持续运行以保持容器活跃
sleep infinity
