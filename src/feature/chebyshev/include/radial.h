#ifndef RADIAL_H
#define RADIAL_H

#include <cmath>
#include <iostream>
#include <string.h>
#include <algorithm>
#include "basic.h"

using namespace std;

template <typename CoordType>
class SmoothFunc {     // smooth switching function
public:
    SmoothFunc();

    SmoothFunc(CoordType rcut_max, CoordType rcut_smooth);

    SmoothFunc(const SmoothFunc& other);

    CoordType get_smooth(CoordType rij) const; // get smooth switching function

    CoordType get_dsmooth(CoordType rij) const; // get partial derivative of smooth switching function with respect to rij

    CoordType rcut_max;
    CoordType rcut_smooth;
private:
    // CoordType rcut_max;
    // CoordType rcut_smooth;
};   // end of class SwithFunc


template <typename CoordType>
class Radial { // radial basis functions
public:
    Radial();

    Radial(int mu, int beta, int ntypes, CoordType rcut_max, CoordType rcut_smooth);

    Radial(const Radial& other);

    ~Radial();

    void build(CoordType rij, int itype, int jtype); // build radial basis functions

    const int get_mu() const; // get number of radial basis functions

    void set_c(CoordType* c);  // set parameters for radial basis functions

    CoordType*** get_rads() const; // get radial basis functions

    CoordType*** get_drads() const; // get partial derivative of radial basis functions with respect to rij

    CoordType**** get_drads2c() const; // get partial derivative of radial basis functions with respect to c

    CoordType**** get_ddrads2c() const; // get partial derivative of drads with respect to c

    void show() const; // show radial basis functions

private:
    int beta, mu, ntypes;
    SmoothFunc<CoordType> smooth;
    Chebyshev1st<CoordType> chebyshev;
    CoordType*** rads;    // radial basis functions    
    CoordType*** drads;    // partial derivative of radial basis functions with respect to rij
    CoordType**** drads2c;    // partial derivative of radial basis functions with respect to c
    CoordType**** ddrads2c;    // partial derivative of drads with respect to c
    CoordType* c;   // parameters for radial basis functions
};   // end of class Radial


/*
below is the implementation of the template classes
*/

template <typename CoordType>
SmoothFunc<CoordType>::SmoothFunc() {};  // default constructor

/**
 * @brief Constructor for the SmoothFunc class.
 * 
 * @param rcut_max The maximum cutoff radius.
 * @param rcut_smooth The smoothing cutoff radius.
 * @tparam CoordType The type of the coordinates.
 */
template <typename CoordType>
SmoothFunc<CoordType>::SmoothFunc(CoordType rcut_max, CoordType rcut_smooth) {
    this->rcut_max = rcut_max;
    this->rcut_smooth = rcut_smooth;
}  // constructor

/**
 * @brief Copy constructor for the SmoothFunc class.
 */
template <typename CoordType>
SmoothFunc<CoordType>::SmoothFunc(const SmoothFunc& other) {
    this->rcut_max = other.rcut_max;
    this->rcut_smooth = other.rcut_smooth;
}  // copy constructor

/**
 * @brief Get the smooth switching function.
 * 
 * @param rij The sqrt(sum(distance**2)) between two atoms.
 */
template <typename CoordType>
CoordType SmoothFunc<CoordType>::get_smooth(CoordType rij) const {
    CoordType fc;
    // CoordType uu = (rij - this->rcut_smooth) / (this->rcut_max - this->rcut_smooth);

    if (rij < this->rcut_smooth) {
        fc = 1.0;
    } else if (rij >= this->rcut_smooth && rij < this->rcut_max) {
        // fc = std::pow(uu, 3) * (-6.0 * std::pow(uu, 2) + 15.0 * uu - 10.0) + 1.0;
        fc = 0.5 * (std::cos(M_PI * (rij - this->rcut_smooth) / (this->rcut_max - this->rcut_smooth)) + 1.0);
    } else {
        fc = 0.0;
    }
    return fc;
}  // get_smooth

/**
 * @brief Get the partial derivative of the smooth switching function with respect to rij.
 */
template <typename CoordType>
CoordType SmoothFunc<CoordType>::get_dsmooth(CoordType rij) const {
    CoordType dfc;
    // CoordType uu = (rij - this->rcut_smooth) / (this->rcut_max - this->rcut_smooth);

    if (rij < this->rcut_smooth) {
        dfc = 0.0;
    } else if (rij >= this->rcut_smooth && rij < this->rcut_max) {
        // dfc = 1.0 / (this->rcut_max - this->rcut_smooth) * (-30.0 * std::pow(uu, 4) + 60.0 * std::pow(uu, 3) - 30.0 * std::pow(uu, 2));
        dfc = -0.5 * M_PI / (this->rcut_max - this->rcut_smooth) * std::sin(M_PI * (rij - this->rcut_smooth) / (this->rcut_max - this->rcut_smooth));
    } else {
        dfc = 0.0;
    }
    return dfc;
}  // get_dsmooth


template <typename CoordType>
Radial<CoordType>::Radial() {};  // default constructor

/**
 * @brief Constructor for the Radial class.
 * 
 * @param mu The number of the radial basis functions.
 * @param beta The order of the Chebyshev polynomials.
 * @param ntypes The number of atom types.
 * @param rcut_max The maximum cutoff radius.
 * @param rcut_smooth The smoothing cutoff radius.
 * @tparam CoordType The type of the coordinates.
 */
template <typename CoordType>
Radial<CoordType>::Radial(int mu, int beta, int ntypes, CoordType rcut_max, CoordType rcut_smooth)
    : mu(mu), beta(beta), ntypes(ntypes), smooth(rcut_max, rcut_smooth), chebyshev(beta, rcut_max, rcut_smooth) {
    // this->rads = new CoordType[mu];
    // std::fill_n(this->rads, mu, CoordType());
    // this->drads = new CoordType[mu];
    // std::fill_n(this->drads, mu, CoordType());
    this->rads = new CoordType**[ntypes];
    this->drads = new CoordType**[ntypes];
    this->drads2c = new CoordType***[ntypes];
    this->ddrads2c = new CoordType***[ntypes];
    for (int i = 0; i < ntypes; i++) {
        this->rads[i] = new CoordType*[ntypes];
        this->drads[i] = new CoordType*[ntypes];
        this->drads2c[i] = new CoordType**[ntypes];
        this->ddrads2c[i] = new CoordType**[ntypes];
        for (int j = 0; j < ntypes; j++) {
            this->rads[i][j] = new CoordType[mu];
            std::fill_n(this->rads[i][j], mu, CoordType());
            this->drads[i][j] = new CoordType[mu];
            std::fill_n(this->drads[i][j], mu, CoordType());
            this->drads2c[i][j] = new CoordType*[mu];
            this->ddrads2c[i][j] = new CoordType*[mu];
            for (int k = 0; k < mu; k++) {
                this->drads2c[i][j][k] = new CoordType[beta];
                std::fill_n(this->drads2c[i][j][k], beta, CoordType());
                this->ddrads2c[i][j][k] = new CoordType[beta];
                std::fill_n(this->ddrads2c[i][j][k], beta, CoordType());
            }
        }
    }

    
}  // constructor

/**
 * @brief Copy constructor for the Radial class.
 */
template <typename CoordType>
Radial<CoordType>::Radial(const Radial& other)
    : mu(other.mu), beta(other.beta), smooth(other.smooth), chebyshev(other.chebyshev) {
    // this->rads = new CoordType[mu];
    // std::copy_n(other.rads, mu, this->rads);
    // this->drads = new CoordType[mu];
    // std::copy_n(other.drads, mu, this->drads);
    this->rads = new CoordType**[ntypes];
    this->drads = new CoordType**[ntypes];
    this->drads2c = new CoordType***[ntypes];
    this->ddrads2c = new CoordType***[ntypes];
    for (int i = 0; i < ntypes; i++) {
        this->rads[i] = new CoordType*[ntypes];
        this->drads[i] = new CoordType*[ntypes];
        this->drads2c[i] = new CoordType**[ntypes];
        this->ddrads2c[i] = new CoordType**[ntypes];
        for (int j = 0; j < ntypes; j++) {
            this->rads[i][j] = new CoordType[mu];
            std::copy_n(other.rads[i][j], mu, this->rads[i][j]);
            this->drads[i][j] = new CoordType[mu];
            std::copy_n(other.drads[i][j], mu, this->drads[i][j]);
            this->drads2c[i][j] = new CoordType*[mu];
            this->ddrads2c[i][j] = new CoordType*[mu];
            for (int k = 0; k < mu; k++) {
                this->drads2c[i][j][k] = new CoordType[beta];
                std::copy_n(other.drads2c[i][j][k], beta, this->drads2c[i][j][k]);
                this->ddrads2c[i][j][k] = new CoordType[beta];
                std::copy_n(other.ddrads2c[i][j][k], beta, this->ddrads2c[i][j][k]);
            }
        }
    }

}  // copy constructor

/**
 * @brief Destructor for the Radial class.
 */
template <typename CoordType>
Radial<CoordType>::~Radial() {
    // for (int i = 0; i < this->mu; i++) {
    //     delete[] this->c[i];
    // }
    for (int i = 0; i < this->ntypes; i++) {
        for (int j = 0; j < this->ntypes; j++) {
            delete[] this->rads[i][j];
            delete[] this->drads[i][j];
            for (int k = 0; k < this->mu; k++) {
                delete[] this->drads2c[i][j][k];
                delete[] this->ddrads2c[i][j][k];
            }
            delete[] this->drads2c[i][j];
            delete[] this->ddrads2c[i][j];
        }
        delete[] this->rads[i];
        delete[] this->drads[i];
        delete[] this->drads2c[i];
        delete[] this->ddrads2c[i];
    }

    delete[] this->rads;
    delete[] this->drads;
    delete[] this->drads2c;
    delete[] this->ddrads2c;
    delete[] this->c;

}  // destructor

/**
 * @brief Build radial basis functions.
 * 
 * @param rij The distance between two atoms.
 * @param itype The type of the i-th atom.
 * @param jtype The type of the neighbor atom.
 */
template <typename CoordType>
void Radial<CoordType>::build(CoordType rij, int itype, int jtype) {
    CoordType fc = this->smooth.get_smooth(rij);
    CoordType dfc = this->smooth.get_dsmooth(rij);
    CoordType rcut_max = this->smooth.rcut_max;
    CoordType rcut_smooth = this->smooth.rcut_smooth;
    Chebyshev1st<CoordType> cheb = this->chebyshev;
    cheb.build(rij);
    // cheb.show();
    const CoordType* vals = cheb.get_vals();
    const CoordType* ders2r = cheb.get_ders2r();
    for (int m = 0; m < this->mu; m++) {
        this->rads[itype][jtype][m] = 0.0;
        this->drads[itype][jtype][m] = 0.0;
        for (int ii = 0; ii < this->beta; ii++) {
            int index = itype * this->ntypes * this->mu * this->beta + jtype * this->mu * this->beta + m * this->beta + ii;
            // std::cout << "index: " << index << std::endl;
            // this->rads[itype][jtype][m] += vals[ii] * fc * this->c[index];     // \sum_{i=0}^{beta-1} c* vals[i] * fc, vals[i] is the i-th Chebyshev polynomial value
            // this->drads[itype][jtype][m] += (ders2r[ii] * fc + vals[ii] * dfc) * this->c[index];     // \sum_{i=0}^{beta-1} c * ders2r[i] * fc + c * vals[i] * dfc, ders2r[i] is the i-th Chebyshev polynomial derivative with respect to rij
            // this->drads2c[itype][jtype][m][ii] = vals[ii] * fc;
            // this->ddrads2c[itype][jtype][m][ii] = (ders2r[ii] * fc + vals[ii] * dfc);
            this->rads[itype][jtype][m] += 0.5 * (vals[ii] * (2.0 * std::pow((rij - rcut_smooth) / (rcut_max - rcut_smooth) - 1.0, 2) - 1.0) + 1.0) * fc * this->c[index];
            this->drads[itype][jtype][m] += 0.5 * this->c[index] * ((ders2r[ii] * (2.0 * std::pow((rij - rcut_smooth) / (rcut_max - rcut_smooth) - 1.0, 2) - 1.0) + \
                                            vals[ii] * 4.0 * ((rij - rcut_smooth) / (rcut_max - rcut_smooth) - 1.0) * (1.0 / (rcut_max - rcut_smooth))) * fc + \
                                            0.5 * (vals[ii] * (2.0 * std::pow((rij - rcut_smooth) / (rcut_max - rcut_smooth) - 1.0, 2) - 1.0) + 1.0) * dfc);
            this->drads2c[itype][jtype][m][ii] = 0.5 * (vals[ii] * (2.0 * std::pow((rij - rcut_smooth) / (rcut_max - rcut_smooth) - 1.0, 2) - 1.0) + 1.0) * fc;
            this->ddrads2c[itype][jtype][m][ii] = 0.5 * (ders2r[ii] * (2.0 * std::pow((rij - rcut_smooth) / (rcut_max - rcut_smooth) - 1.0, 2) - 1.0) + \
                                            vals[ii] * 4.0 * ((rij - rcut_smooth) / (rcut_max - rcut_smooth) - 1.0) * (1.0 / (rcut_max - rcut_smooth)) + \
                                            0.5 * (vals[ii] * (2.0 * std::pow((rij - rcut_smooth) / (rcut_max - rcut_smooth) - 1.0, 2) - 1.0) + 1.0) * dfc);
        }
    }


}  // build

/**
 * @brief Get the number of the radial basis functions.
 */
template <typename CoordType>
const int Radial<CoordType>::get_mu() const {
    return this->mu;
}  // get_mu

template <typename CoordType>
void Radial<CoordType>::set_c(CoordType* c) {
    // this->c = new CoordType*[this->mu];
    // for (int i = 0; i < this->mu; i++) {
    //     this->c[i] = new CoordType[this->beta];
    //     for (int j = 0; j < this->beta; j++) {
    //         this->c[i][j] = c[i][j];
    //     }
    // }
    int size = this->ntypes * this->ntypes * this->mu * this->beta;
    this->c = new CoordType[size];
    std::copy_n(c, size, this->c);
}

/**
 * @brief Get the radial basis functions.
 */
template <typename CoordType>
CoordType*** Radial<CoordType>::get_rads() const {
    return this->rads;
}  // get_rads

/**
 * @brief Get the partial derivative of the radial basis functions with respect to rij.
 */
template <typename CoordType>
CoordType*** Radial<CoordType>::get_drads() const {
    return this->drads;
}  // get_drads

/**
 * @brief Get the partial derivative of the radial basis functions with respect to c.
 */
template <typename CoordType>
CoordType**** Radial<CoordType>::get_drads2c() const {
    return this->drads2c;
}  // get_drads2c

/**
 * @brief Get the partial derivative of the partial derivative of the radial basis functions with respect to c.
 */
template <typename CoordType>
CoordType**** Radial<CoordType>::get_ddrads2c() const {
    return this->ddrads2c;
}  // get_ddrads2c


/**
 * @brief Show radial basis functions.
 */
template <typename CoordType>
void Radial<CoordType>::show() const {
    cout << "radial basis functions: ";
    for(int i = 0; i < this->ntypes; i++) {
        for (int j = 0; j < this->ntypes; j++) {
            for (int ii = 0; ii < this->mu; ii++) {
                cout << this->rads[i][j][ii] << " ";
            }
            cout << endl;
        }
    }
    
    cout << "partial radial w.r.t rij: ";
    for(int i = 0; i < this->ntypes; i++) {
        for (int j = 0; j < this->ntypes; j++) {
            for (int ii = 0; ii < this->mu; ii++) {
                cout << this->drads[i][j][ii] << " ";
            }
            cout << endl;
        }
    }

    cout << "partial radial w.r.t c: ";
    for(int i = 0; i < this->ntypes; i++) {
        for (int j = 0; j < this->ntypes; j++) {
            for (int ii = 0; ii < this->mu; ii++) {
                for (int jj = 0; jj < this->beta; jj++) {
                    cout << this->drads2c[i][j][ii][jj] << " ";
                }
                cout << endl;
            }
        }
    }
}  // show


#endif // RADIAL_H