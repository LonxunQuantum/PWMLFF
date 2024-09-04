#!/bin/bash

# 检查 CUDA 版本是否为 11.8
check_cuda_version() {
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d"," -f1 | cut -c2-)
        major_version=$(echo $cuda_version | cut -d'.' -f1)
        minor_version=$(echo $cuda_version | cut -d'.' -f2)

        if [ "$major_version" == "11" ] && [ "$minor_version" == "8" ]; then
            echo "1. CUDA version is 11.8."
        else
            echo "1. CUDA version is not 11.8, current version is $cuda_version."
        fi
    else
        echo "1. nvcc command not found, CUDA might not be installed."
    fi
}

# 检查是否存在 nvcc 命令
check_nvcc() {
    if command -v nvcc &> /dev/null; then
        echo "2. nvcc command exists."
    else
        echo "2. nvcc command does not exist."
    fi
}

# 检查 ifort 编译器版本是否不小于19.1 和 MKL 库是否存在
check_ifort_mkl() {
    if command -v ifort &> /dev/null; then
        ifort_version=$(ifort --version | grep "ifort" | awk '{print $3}' | cut -d'.' -f1-2)
        major_version=$(echo $ifort_version | cut -d'.' -f1)
        minor_version=$(echo $ifort_version | cut -d'.' -f2)
        
        if [ "$major_version" -gt 19 ] || ([ "$major_version" -eq 19 ] && [ "$minor_version" -ge 1 ]); then
            echo "3. ifort version is no less than 19.1, current version is $ifort_version."
        else
            echo "3. ifort version is not greater than 19.1, current version is $ifort_version."
        fi
    else
        echo "3. ifort compiler not found."
    fi
    
    # 检查 MKL 库是否存在
    if [ -d "/opt/intel/mkl" ] || [ -d "$MKLROOT" ]; then
        echo "4. MKL library is installed."
    else
        echo "4. MKL library is not installed."
    fi
}

# 检查 GCC 版本是否为 8.x
check_gcc_version() {
    if command -v gcc &> /dev/null; then
        gcc_version=$(gcc -dumpversion)
        if [[ $gcc_version == 8.* ]]; then
            echo "5. GCC version is 8.x, current version is $gcc_version."
        else
            echo "5. GCC version is not 8.x, current version is $gcc_version."
        fi
    else
        echo "5. GCC not found."
    fi
}

# 检查当前 Python 环境中是否安装了 PyTorch
check_pytorch_installed() {
    python -c "import torch" 2> /dev/null
    if [ $? -eq 0 ]; then
        echo "6. PyTorch is installed."
    else
        echo "6. PyTorch is not installed in the current Python environment."
        return 1
    fi
}

# 检查 PyTorch 版本是否为 2.0 及以上
check_pytorch_version() {
    pytorch_version=$(python -c "import torch; print(torch.__version__)" | cut -d'.' -f1,2)
    major_version=$(echo $pytorch_version | cut -d'.' -f1)
    minor_version=$(echo $pytorch_version | cut -d'.' -f2)
    
    if [ "$major_version" -ge 2 ]; then
        echo "7. PyTorch version is 2.0 or above, current version is $pytorch_version."
    else
        echo "7. PyTorch version is below 2.0, current version is $pytorch_version."
    fi
}

# 执行检查
check_cuda_version
check_nvcc
check_ifort_mkl
check_gcc_version

# 检查 PyTorch 相关信息
check_pytorch_installed && check_pytorch_version

