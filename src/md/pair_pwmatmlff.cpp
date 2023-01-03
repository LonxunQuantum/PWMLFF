#include "pair_pwmatmlff.h"
#include "diy.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
//#include "respa.h"
#include "update.h"
#include "domain.h"

#include <cmath>
#include <cstring>
// tmp
#include <cstdio>

using namespace LAMMPS_NS;
using namespace MathConst;
PairQCAD::PairQCAD(LAMMPS *lmp) : Pair(lmp)
{
    writedata = 1;
}
PairQCAD::~PairQCAD()
{
    if (copymode) return;
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(scale);
        memory->destroy(x_cart);
        memory->destroy(x_cart_tmp);
        memory->destroy(x_frac);
        memory->destroy(f_atom);
        memory->destroy(itype_atom);
        memory->destroy(itype_tmp);
        memory->destroy(e_atom);
  }
}
void PairQCAD::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair_pwmatmlff:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;
  memory->create(cutsq,n,n,"pair_pwmatmlff:cutsq");
  memory->create(scale,n,n,"pair_pwmatmlff:scale");

  memory->create(e_atom, atom->natoms, "pair_pwmatmlff:e_atom");
  memory->create(x_cart, atom->natoms, 3, "pair_pwmatmlff:x_cart");
  memory->create(x_cart_tmp, atom->natoms, 3, "pair_pwmatmlff:x_cart_tmp");
  memory->create(x_frac, atom->natoms, 3, "pair_pwmatmlff:x_frac");
  memory->create(f_atom, atom->natoms, 3, "pair_pwmatmlff:f_atom");
  memory->create(itype_atom, atom->natoms, "pair_pwmatmlff:itype_atom");
  memory->create(itype_tmp, atom->natoms, "pair_pwmatmlff:itype_tmp");
}

void PairQCAD::compute(int eflag, int vflag)
{
  int myrank, nprocs;
  MPI_Status p_mpi_status;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  num_atoms = atom->natoms;
  // `double* eatom;` in pair.h
  //double* e_atom = eatom;

  // get lattice (deepmd dbox)
  // a1 = (0 1 2)
  // a2 = (3 4 5)
  // a3 = (6 7 8)
  lattice[0] = domain->h[0];   // xx
  lattice[4] = domain->h[1];   // yy
  lattice[8] = domain->h[2];   // zz
  lattice[7] = domain->h[3];   // yz
  lattice[6] = domain->h[4];   // xz
  lattice[3] = domain->h[5];   // xy
  // get cartesian coordinates
  //printf("myrank, natoms local: %d, %d\n", myrank, atom->nlocal);
  // max num of processes = 1000
  int all_natomlocals[1000];              // 3 2 2 2
  int accumulated_natomlocals[1000];      // 3 5 7 9
  if (nprocs > 1000)    error->all(FLERR, "too many processes, modify pair_pwmatmlff.cpp for more processes");

  if (myrank != 0)
  {
    MPI_Send(&atom->nlocal, 1, MPI_INT, 0, 101, MPI_COMM_WORLD);
    MPI_Send(&atom->x[0][0], 3*atom->nlocal, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD);
    MPI_Send(&atom->type[0], atom->nlocal, MPI_INT, 0, 103, MPI_COMM_WORLD);
  }
  else // rank0
  {
    all_natomlocals[0] = atom->nlocal;
    accumulated_natomlocals[0] = all_natomlocals[0];
    // coords in rank0
    for (int i = 0; i < atom->nlocal; i++)  // i for atom id
    {
        for (int j = 0; j < 3; j++)
        {
            x_cart_tmp[i][j] = atom->x[i][j];
            itype_tmp[i] = atom->type[i];
        }
        // printf("llptest333, x_cart_tmp: %f %f %f\n", atom->x[i][0], atom->x[i][1], atom->x[i][2]);
    }
    for (int i = 1; i < nprocs; i++)  // i for other ranks
    {
        int tmp_local;
        MPI_Recv(&tmp_local, 1, MPI_INT, i, 101, MPI_COMM_WORLD, &p_mpi_status);
        all_natomlocals[i] = tmp_local;
        accumulated_natomlocals[i] = accumulated_natomlocals[i-1] + all_natomlocals[i];
        MPI_Recv(&x_cart_tmp[accumulated_natomlocals[i-1]][0], 3*all_natomlocals[i], MPI_DOUBLE, i, 102, MPI_COMM_WORLD, &p_mpi_status);
        MPI_Recv(&itype_tmp[accumulated_natomlocals[i-1]], all_natomlocals[i], MPI_INT, i, 103, MPI_COMM_WORLD, &p_mpi_status);
    }
  }
  // rank0 unique op end
  // broadcast x_cart_tmp to all other processes
  MPI_Bcast(&x_cart_tmp[0][0], 3*atom->natoms, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&accumulated_natomlocals[0], nprocs, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&itype_tmp[0], atom->natoms, MPI_INT, 0, MPI_COMM_WORLD);
  // 
//if (myrank == 1)
//{
//  printf("type_map: %d    %d\n", type_map[0], type_map[1]);
//  for (int i = 0; i < atom->natoms; i++) printf("itype_tmp: %d\n", itype_tmp[i]);
//}
  for (int i = 0; i < atom->natoms; i++)
  {
    itype_atom[i] = type_map[itype_tmp[i]-1]; // atom->type starts from 1 in lammps.data, and type_map index starts from 0;
    for (int j = 0; j < 3; j++)
    {
        x_cart[i][j] = x_cart_tmp[i][j] - domain->boxlo[j];
    }
  }
  // fortran-mkl-dgemm needed pointers
  for(int i = 0; i < 9; i++) reclat[i] = lattice[i];
  diy::dinv(&reclat[0],3);
  int m_dg = 1; int n_dg = 3; int k_dg = 3; int lda = 1; int ldb = 3; int ldc = 1; char transa = 'N'; char transb = 'N'; double alpha_dg = 1.0; double beta_dg = 0.0;
  // convert cart to frac
  for(int i = 0; i < num_atoms; i++)
  {
      //dgemm_('N', 'N', 1, 3, 3, 1.0, &x_cart[i][0], 1, &reclat[0][0], 3, 0.0, &tmp_v3[0], 1);
      dgemm_(&transa, &transb, &m_dg, &n_dg, &k_dg, &alpha_dg, &x_cart[i][0], &lda, &reclat[0], &ldb, &beta_dg, &tmp_v3[0], &ldc);
      x_frac[i][0] = tmp_v3[0];
      x_frac[i][1] = tmp_v3[1];
      x_frac[i][2] = tmp_v3[2];
  }
  // call fortran subroutine energy force
  if (iago >= ievery)
  {
    iflag_reneighbor = 1;
    iago = 0;
  }
  iago ++;
  
  /*
    ff core
  */
  f2c_calc_energy_force(&imodel, &num_atoms, &itype_atom[0], &lattice[0], &x_frac[0][0], &e_atom[0], &f_atom[0][0], &e_tot, &iflag_reneighbor);
  
  // f_atom(all forces) to atom->f
  //printf("llptest6\n");
  int local_position = accumulated_natomlocals[myrank] - atom->nlocal; 
  //printf("my rank, accumulated_natomlocals: %d %d\n", myrank, accumulated_natomlocals[myrank]);
  for (int i = 0; i < atom->nlocal; i++)
  {
    for (int j = 0; j < 3; j++)
    {
        atom->f[i][j] = f_atom[i+local_position][j]; //
    }
  }

  // potential energy
  if (eflag)
  {
    eng_vdwl = e_tot * 1.0 / nprocs; // this line may conflict with other force fields
  }

}

void PairQCAD::settings(int argc, char** argv)
{
}

void PairQCAD::coeff(int narg, char** arg)
{
  // check argv
//for (int i = 0; i < narg; i++)
//{
//  printf("pair_pwmatmlff, args. arg[%d]: %s\n", i, arg[i]);
//}
  if (!allocated) allocate();
  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }
  //printf("llptest999, type_map:\n");
  // pair_coeff * *     1         29  8 ....
  // pair_coeff * * linear_model  Cu  O ....
  if (narg-4 < atom->ntypes)
    error->one(FLERR, "pair_coeff error. add imodel and type map to pair_coeff");
  imodel = atoi(arg[2]);
  ievery = atoi(arg[3]);
  for (int i = 0; i < narg - 4; i++)
  {
    type_map[i] = atoi(arg[i+4]);
  }
}
/* units metal
For style metal, these are the units:
    mass = grams/mole
    distance = Angstroms
    time = picoseconds
    energy = eV
    velocity = Angstroms/picosecond
    force = eV/Angstrom
    torque = eV
    temperature = Kelvin
    pressure = bars
    dynamic viscosity = Poise
    charge = multiple of electron charge (1.0 is a proton)
    dipole = charge*Angstroms
    electric field = volts/Angstrom
    density = gram/cm^dim
*/
