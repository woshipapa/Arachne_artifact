#!/bin/bash

# 检查是否提供了提交信息作为参数
if [ -z "$1" ]; then
  # 如果没有提供，就打印错误信息并退出
  echo "❌ 错误: 请提供提交信息！"
  echo "用法: ./push_flex.sh \"你的提交信息\""
  exit 1
fi

# 将第一个参数赋值给 COMMIT_MESSAGE 变量，更清晰
COMMIT_MESSAGE="$1"

# --- 开始执行Git命令 ---

echo "➡️ 1/3: 正在添加所有文件 (git add .)"
git add .

echo "➡️ 2/3: 正在提交 (git commit)"
git commit -m "$COMMIT_MESSAGE"

echo "➡️ 3/3: 正在推送到 origin/dynamic-flex (git push)"
git push -u origin dynamic-flex

echo "✅ 完成！"