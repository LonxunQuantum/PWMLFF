#!/bin/bash

mkdir bin
mkdir lib
make -C pre_data/gen_feature
make -C pre_data/fit
make -C pre_data/fortran_code  # spack load gcc@7.5.0

make -C md/fortran_code
# make nep find_neigh interface
cd feature/nep_find_neigh
rm -rf build/*
rm findneigh.so
mkdir build
cd build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && make
cp findneigh.* ../findneigh.so
cd ..
cd ..
cd ..
# compile gpumd
#make -C GPUMD/src

cd bin

ln -s ../pre_data/mlff.py .
ln -s ../pre_data/seper.py .
ln -s ../pre_data/gen_data.py .
ln -s ../pre_data/data_loader_2type.py .
ln -s ../../utils/read_torch_wij.py . 
ln -s ../../utils/plot_nn_test.py . 
ln -s ../../utils/plot_mlff_inference.py .
ln -s ../../utils/read_torch_wij_dp.py . 
ln -s ../md/fortran_code/main_MD.x .
ln -s ../../pwmlff_main.py ./PWMLFF
ln -s ../../pwdata_main.py pwdata

#ln -s ../GPUMD/src/gpumd .
#ln -s ../GPUMD/src/nep .

chmod +x ./mlff.py
chmod +x ./seper.py
chmod +x ./gen_data.py
chmod +x ./data_loader_2type.py
chmod +x ./train.py
chmod +x ./test.py
chmod +x ./predict.py
chmod +x ./read_torch_wij.py
chmod +x ./read_torch_wij_dp.py
chmod +x ./plot_nn_test.py
chmod +x ./plot_mlff_inference.py 

cd ..            # back to src dir

################################################################
#Writing environment variable

current_path=$(pwd)

if ! grep -q "^export PYTHONPATH=.*$current_path" ~/.bashrc; then
  echo "export PYTHONPATH=$current_path:\$PYTHONPATH" >> ~/.bashrc
fi

if ! grep -q "^export PATH=.*$current_path/bin" ~/.bashrc; then
  echo "export PATH=$current_path/bin:\$PATH" >> ~/.bashrc
fi
##################################################################
cd op
rm build -r
# python setup.py install --user
mkdir build
cd build
cmake ..
make
echo "Environment variables have been written ~/.bashrc"
