CALCFTYPE1=gen_2b_feature.f90 find_neighbore00.f90 find_neighbore.f90 find_feature_2b_type1.f90 find_feature_2b_type2.f90 find_feature_2b_type3.f90 gaussj.f get_ALI.f
CALCFTYPE2=gen_3b_feature.f90 find_neighbore00.f90 find_neighbore.f90 find_feature_3b_type1.f90 find_feature_3b_type2.f90 find_feature_3b_type3.f90 gaussj.f get_ALI.f
CALCLINSRC=calc_lin_forceM_module.f90
MDVV=md_VV.f90
DFEAT=read_dfeatNN.f90
CALCREP=calc_replusive.f90
READALL=read_all.f90
CONVDFEAT=convert_dfeat.f90
CONVDFEAT64=convert_dfeat64.f90
lib_path=../../lib
.PHONY: py


py: calc_ftype1.so calc_ftype2.so calc_lin.so calc_vv.so read_dfeatNN.so calc_rep.so read_all.so convert_dfeat.so convert_dfeat64.so
	mv *.so $(lib_path)

calc_ftype1.so: $(CALCFTYPE1)
	python3 -m numpy.f2py -c -m calc_ftype1 --fcompiler=intelem --compiler=intelem -L$(MKLROOT)/lib/intel64/ -lmkl_rt $(CALCFTYPE1)
	mv calc_ftype1.cpython-*-x86_64-linux-gnu.so calc_ftype1.so

calc_ftype2.so: $(CALCFTYPE2)
	python3 -m numpy.f2py -c -m calc_ftype2 --fcompiler=intelem --compiler=intelem -L$(MKLROOT)/lib/intel64/ -lmkl_rt $(CALCFTYPE2)
	mv calc_ftype2.cpython-*-x86_64-linux-gnu.so calc_ftype2.so

calc_lin.so: $(CALCLINSRC)
	python3 -m numpy.f2py -c -m calc_lin --fcompiler=intelem --compiler=intelem -L$(MKLROOT)/lib/intel64/ -lmkl_rt $(CALCLINSRC)
	mv calc_lin.cpython-*-x86_64-linux-gnu.so calc_lin.so

calc_vv.so: $(MDVV)
	python3 -m numpy.f2py -c -m calc_vv --fcompiler=intelem --compiler=intelem -L$(MKLROOT)/lib/intel64/ -lmkl_rt $(MDVV)
	mv calc_vv.cpython-*-x86_64-linux-gnu.so calc_vv.so

read_dfeatNN.so: $(DFEAT)
	python3 -m numpy.f2py -c -m read_dfeatNN --fcompiler=intelem --compiler=intelem -L$(MKLROOT)/lib/intel64/ -lmkl_rt $(DFEAT)
	mv read_dfeatNN.cpython-*-x86_64-linux-gnu.so read_dfeatNN.so

calc_rep.so: $(CALCREP)
	python3 -m numpy.f2py -c -m calc_rep --fcompiler=intelem --compiler=intelem -L$(MKLROOT)/lib/intel64/ -lmkl_rt $(CALCREP)
	mv calc_rep.cpython-*-x86_64-linux-gnu.so calc_rep.so

read_all.so: $(READALL)
	python3 -m numpy.f2py -c -m read_all --fcompiler=intelem --compiler=intelem -L$(MKLROOT)/lib/intel64/ -lmkl_rt $(READALL)
	mv read_all.cpython-*-x86_64-linux-gnu.so read_all.so

convert_dfeat.so: $(CONVDFEAT)
	python3 -m numpy.f2py -c -m convert_dfeat --fcompiler=intelem --compiler=intelem -L$(MKLROOT)/lib/intel64/ -lmkl_rt $(CONVDFEAT)
	mv convert_dfeat.cpython-*-x86_64-linux-gnu.so convert_dfeat.so

convert_dfeat64.so: $(CONVDFEAT64)
	python3 -m numpy.f2py -c -m convert_dfeat64 --fcompiler=intelem --compiler=intelem -L$(MKLROOT)/lib/intel64/ -lmkl_rt $(CONVDFEAT64)
	mv convert_dfeat64.cpython-*-x86_64-linux-gnu.so convert_dfeat64.so

clean:
	rm -f *.so $(lib_path)/*.so
