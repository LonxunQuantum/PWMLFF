#!/bin/bash

mkdir bin
mkdir lib
make -C pre_data/gen_feature
make -C pre_data/fit
make -C pre_data/fortran_code  # spack load gcc@7.5.0

make -C md/fortran_code
cd bin

ln -s ../pre_data/mlff.py .
ln -s ../pre_data/seper.py .
ln -s ../pre_data/gen_data.py .
ln -s ../pre_data/data_loader_2type.py .
ln -s ../../utils/read_torch_wij.py . 
ln -s ../../utils/plot_nn_test.py . 
ln -s ../../utils/plot_mlff_inference.py .
ln -s ../../utils/read_torch_wij_dp.py . 
ln -s ../../utils/outcar2movement . 

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
chmod +x ./outcar2movement

cd ..            # back to src dir

cd op
python setup.py install 
