#!/bin/bash

# 检查是否提供了参数
if [ -z "$1" ]; then
  echo "请提供一个字符串作为参数"
  exit 1
fi

# 获取参数
X=$1

# 查找并杀死包含指定字符串的进程
pids=$(ps aux | grep "$X" | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
  echo "没有找到包含 '$X' 的进程"
else
  echo "正在杀死以下进程:"
  echo "$pids"
  kill $pids
fi