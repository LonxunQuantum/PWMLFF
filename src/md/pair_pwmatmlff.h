/* -*- c++ -*- ----------------------------------------------------------
    liuliping, PWmat-MLFF to LAMMPS
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pwmatmlff,PairQCAD);
// clang-format on
#else

#ifndef LMP_PAIR_QCAD_H
#define LMP_PAIR_QCAD_H

#include "pair.h"

namespace LAMMPS_NS {

class PairQCAD : public Pair {
 public:
  PairQCAD(class LAMMPS *);
  ~PairQCAD() override;
  int num_atoms;
  int ievery = 1000;
  int iago = ievery;
  int imodel = 1; // should be read from model file
  int iflag_reneighbor = 1;
  double** x_cart;
  double** x_cart_tmp;
  double** x_frac;
  double** f_atom; // x_cart and f_atom are stored seperaterly in different processes.
  double** scale;
  int* itype_atom;
  int* itype_tmp;
  double* e_atom;
  double lattice[9];
  double reclat[9];
  double tmp_v3[3];
  double e_tot;

  // read type_map from pair_coeff of lmp.in
  // max 10 elements, this array should be [29, 8, 0,...] for CuO
  int type_map[10];
  
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  //void init_style() override;
  //double init_one(int, int) override;
  //void write_restart(FILE *) override;
  //void read_restart(FILE *) override;
  //void write_restart_settings(FILE *) override;
  //void read_restart_settings(FILE *) override;
  //void write_data(FILE *) override;
  //void write_data_all(FILE *) override;
  //double single(int, int, int, int, double, double, double, double &) override;
  //void *extract(const char *, int &) override;
  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif

