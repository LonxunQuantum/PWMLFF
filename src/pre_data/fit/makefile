.PHONY: all

all: fit_lin_forceMM.r fit_VV_forceMM.r fit_vdw.r calc_lin_forceMM.r select_mm_VV_forceMPI.r feat_dist_xp.r feat_collect_PCA.r select_mm_VV.r calc_VV_forceMM.r
	\cp *.r ../../bin/

MPIFC = mpiifort
FFLAG = -O3
# mpif90 = intelmpi+ifort on some machine, but they won't use this any more
#is_mpiifort := $(shell which mpiifort 1>&2 2> /dev/null; echo $$?)
#ifneq ($(is_mpiifort),0)
#	MPIFC=mpif90
#endif

%.mod: %.f90
	$(MPIFC) $(FFLAG) -c $<
%.o:  %.f90
	$(MPIFC) $(FFLAG) -c $<
%.o:  %.f
	$(MPIFC) $(FFLAG) -c $<
%.o: %.cpp
	$(MPICXX) $(CFLAG) -c $<

feat_collect_PCA.r: feat_collect_PCA.o ran1.o
	$(MPIFC)  $(FFLAG) -o $@ $^ -mkl
calc_VV_forceMM.r: %.r : %.o
	$(MPIFC)  $(FFLAG) -o $@ $^ -mkl
fit_vdw.r: %.r : %.o
	$(MPIFC)  $(FFLAG) -o $@ $^ -mkl
select_mm_VV.r: select_mm_VV.o ran1.o
	$(MPIFC)  $(FFLAG) -o $@ $^ -mkl
select_mm_VV_G2MPI.r: select_mm_VV_G2MPI.o ran1.o
	$(MPIFC)  $(FFLAG) -o $@ $^ -mkl



calc_lin_forceMM.r: data_ewald.o get_ewald.o get_ALI.o gaussj.o calc_lin_forceMM.o
	$(MPIFC) $(FFLAG) -mkl $^ -o calc_lin_forceMM.r

fit_lin_forceMM.r:  data_ewald.o get_ewald.o get_ALI.o gaussj.o fit_lin_forceMM.o
	$(MPIFC) $(FFLAG) -mkl $^ -o fit_lin_forceMM.r

fit_VV_forceMM.r:  data_ewald.o get_ewald.o get_ALI.o gaussj.o fit_VV_forceMM.o
	$(MPIFC) $(FFLAG) -mkl $^ -o fit_VV_forceMM.r


select_mm_VV_forceMPI.r:
	$(MPIFC) $(FFLAG) select_mm_VV_forceMPI.f90  calc_VV_force_sub.f90 fit_VV_force_sub.f90 -mkl -o $@
feat_dist_xp.r:
	$(MPIFC) $(FFLAG) feat_dist_xp.f90 -mkl -o $@

#$(NORANEXE): %.r : %.o
#	ifort -o $@ $^ -mkl
#$(NORANOBJ): %.o : %.f90
#	ifort -c $^
#$(RANEXE): %.r : %.o ran1.o
#	ifort -o $@ $^ -mkl
#$(RANOBJ): %.o : %.f90
#	ifort -c $^
#ran1.o: ran1.f90
#	ifort -c ran1.f90


clean:
	rm -f *.o *.r *.mod
	rm -f ../../bin/calc_lin_forceMM.r
	rm -f ../../bin/feat_collect_PCA.r
	rm -f ../../bin/fit_lin_forceMM.r
	rm -f ../../bin/fit_VV_forceMM.r
	rm -f ../../bin/select_mm_VV.r
	rm -f ../../bin/calc_VV_forceMM.r
	rm -f ../../bin/feat_dist_xp.r
	rm -f ../../bin/fit_vdw.r
	rm -f ../../bin/select_mm_VV_forceMPI.r
	rm -f ../../bin/select_mm_VV_G2MPI.r

