#!/bin/bash
rm -f bin/*.r
rm -f bin/*.x
rm -f bin/*.py
rm /bin -rf
make clean -C pre_data/gen_feature
make clean -C pre_data/fit
make clean -C pre_data/fortran_code  # for tf, will delete it
make clean -C QCAD/fortran_code

