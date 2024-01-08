all:gen_dR.x gen_3b_feature.x gen_2b_feature.x gen_2bgauss_feature.x gen_3bcos_feature.x gen_MTP_feature.x gen_SNAP_feature.x write_egroup.x gen_deepMD1_feature.x gen_deepMD2_feature.x NeighConst.so
	\cp *.x ../../bin/
	\cp *.so ../../lib/

# FFLAG = -g -traceback
FFLAG = -O3 -I/opt/iotk/src/
MKL = -mkl
# lib_path=../../lib
# .PHONY: py
# LeastSquaresSolver.mod: least_squares.f90
# 	ifort $(FFLAG) $(MKL) -c $<

# gen_egroup.x:
# 	ifort scan_title.f90 transform_to_upper.f90  gen_egroup.f90 \
#         find_neighbore00.f90 find_neighbore.f90  \
#         -o $@

# py: NeighConst.so
# 	mv *.so $(lib_path)

gen_dR.x: 
	ifort -O3 scan_title.f90 transform_to_upper.f90 \
        find_neighbore00.f90 find_neighbore.f90 gen_dR.f90  \
		-o $@

gen_2b_feature.x: 
	ifort least_squares.f90 counts_atom.f90 scan_title.f90 transform_to_upper.f90 get_grid2b_type1.f90 get_grid2b_type2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_2b_type1.f90 \
        find_feature_2b_type2.f90 find_feature_2b_type3.f90 gen_2b_feature.f90  \
		-o $@ $(MKL)

gen_3b_feature.x: 
	ifort least_squares.f90 counts_atom.f90 scan_title.f90 transform_to_upper.f90 get_grid3b_type1.f90 get_grid3b_type2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_3b_type1.f90 \
        find_feature_3b_type2.f90 find_feature_3b_type3.f90 gen_3b_feature.f90 \
		-o $@ $(MKL)

gen_2bgauss_feature.x: 
	ifort least_squares.f90 counts_atom.f90 scan_title.f90 transform_to_upper.f90 get_grid2b_type1.f90 get_grid2b_type2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_2bgauss.f90 \
        gen_2bgauss_feature.f90 \
		-o $@ $(MKL)

gen_3bcos_feature.x: 
	ifort least_squares.f90 counts_atom.f90 scan_title.f90 transform_to_upper.f90 get_grid2b_type1.f90 get_grid2b_type2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_3bcos.f90 \
        gen_3bcos_feature.f90 \
		-o $@ $(MKL)

gen_MTP_feature.x: 
	ifort -O3 least_squares.f90 counts_atom.f90 scan_title.f90 transform_to_upper.f90 get_expand_MT.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_MTP.f90 calc_loop_ind.f90 calc_polynomial.f90 \
        gen_MTP_feature.f90 \
		-o $@ $(MKL)

gen_SNAP_feature.x: 
	ifort -O3 least_squares.f90 counts_atom.f90 scan_title.f90 transform_to_upper.f90 factorial.f90 calc_clebsch_gordan.f90 calc_U_JM1M2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_SNAP.f90  \
        gen_SNAP_feature.f90 \
		-o $@ $(MKL)

gen_deepMD1_feature.x: 
	ifort -O3 least_squares.f90 counts_atom.f90 scan_title.f90 transform_to_upper.f90 calc_chebyshev.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_deepMD1.f90  \
        gen_deepMD1_feature.f90 \
		-o $@ $(MKL)

gen_deepMD2_feature.x: 
	ifort -O3 least_squares.f90 counts_atom.f90 scan_title.f90 transform_to_upper.f90 \
        find_neighbore00.f90 find_neighbore.f90 find_feature_deepMD2.f90  \
        gen_deepMD2_feature.f90 \
		-o $@ $(MKL)

write_egroup.x:
	ifort scan_title.f90 transform_to_upper.f90  write_egroup.f90 \
        find_neighbore00.f90 find_neighbore.f90  \
		-o $@

NeighConst.so: NeighConst.f90
	python3 -m numpy.f2py -c -m NeighConst --fcompiler=intelem --compiler=intelem -L$(MKLROOT)/lib/intel64/ -lmkl_rt NeighConst.f90
	mv NeighConst.cpython-*-x86_64-linux-gnu.so NeighConst.so		

clean:
	rm -f *.x *.mod *.so
