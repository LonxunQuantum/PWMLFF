#!/bin/bash

# 打包环境
#conda pack -n pwmlff-2024.5

# 将打包好的环境和 PWMLFF 目录打包成 tar.gz 文件
tar -czf PWMLFF-2024.5.tar.gz pwmlff-2024.5.tar.gz PWMLFF lammps-2024.5

# 将 tar.gz 文件编码成 base64
base64 PWMLFF-2024.5.tar.gz > PWMLFF-2024.5.tar.gz.base64

# 复制模板脚本并添加 base64 编码的 tar.gz 数据
cp pwmlff.2024.5.sh.template pwmlff-2024.5.sh
cat PWMLFF-2024.5.tar.gz.base64 >> pwmlff-2024.5.sh

# 打包最终的脚本
tar -czvf pwmlff-2024.5.sh.tar.gz pwmlff-2024.5.sh check_offenv.sh

