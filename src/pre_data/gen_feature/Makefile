all: gen_dR.x gen_3b_feature.x gen_2b_feature.x gen_2bgauss_feature.x gen_3bcos_feature.x gen_MTP_feature.x gen_SNAP_feature.x write_egroup.x gen_deepMD1_feature.x gen_deepMD2_feature.x
	\cp *.x ../../bin/

gen_dR.x: 
	ifort -O3 scan_title.f90 transform_to_upper.f90 \
        find_neighbore00.f90 find_neighbore.f90 gen_dR.f90  \
		-o $@

gen_2b_feature.x: 
	ifort scan_title.f90 transform_to_upper.f90 get_grid2b_type1.f90 get_grid2b_type2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_2b_type1.f90 \
        find_feature_2b_type2.f90 find_feature_2b_type3.f90 gen_2b_feature.f90  \
		-o $@

gen_3b_feature.x: 
	ifort scan_title.f90 transform_to_upper.f90 get_grid3b_type1.f90 get_grid3b_type2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_3b_type1.f90 \
        find_feature_3b_type2.f90 find_feature_3b_type3.f90 gen_3b_feature.f90 \
		-o $@

gen_2bgauss_feature.x: 
	ifort scan_title.f90 transform_to_upper.f90 get_grid2b_type1.f90 get_grid2b_type2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_2bgauss.f90 \
        gen_2bgauss_feature.f90 \
		-o $@

gen_3bcos_feature.x: 
	ifort scan_title.f90 transform_to_upper.f90 get_grid2b_type1.f90 get_grid2b_type2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_3bcos.f90 \
        gen_3bcos_feature.f90 \
		-o $@

gen_MTP_feature.x: 
	ifort -O3 scan_title.f90 transform_to_upper.f90 get_expand_MT.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_MTP.f90 calc_loop_ind.f90 calc_polynomial.f90 \
        gen_MTP_feature.f90 \
		-o $@

gen_SNAP_feature.x: 
	ifort -O3  scan_title.f90 transform_to_upper.f90 factorial.f90 calc_clebsch_gordan.f90 calc_U_JM1M2.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_SNAP.f90  \
        gen_SNAP_feature.f90 \
		-o $@

gen_deepMD1_feature.x: 
	ifort -O3 scan_title.f90 transform_to_upper.f90 calc_chebyshev.f90  \
        find_neighbore00.f90 find_neighbore.f90 find_feature_deepMD1.f90  \
        gen_deepMD1_feature.f90 \
		-o $@

gen_deepMD2_feature.x: 
	ifort -O3 scan_title.f90 transform_to_upper.f90 \
        find_neighbore00.f90 find_neighbore.f90 find_feature_deepMD2.f90  \
        gen_deepMD2_feature.f90 \
		-o $@

write_egroup.x:
	ifort scan_title.f90 transform_to_upper.f90  write_egroup.f90 \
        find_neighbore00.f90 find_neighbore.f90  \
		-o $@
clean:
	rm -f *.x *.mod
