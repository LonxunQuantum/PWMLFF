#ifndef BASIC_H
#define BASIC_H

#include <cmath>
#include <iostream>
#include <string.h>
#include <algorithm>

using namespace std;

template <typename CoordType>
class CalcKsi {     // input for chebyshev polynomials
public:
    CalcKsi();

    CalcKsi(CoordType rcut_max, CoordType rcut_smooth);

    CalcKsi(const CalcKsi& other);

    CoordType get_si(CoordType rij) const; // si, input for Chebyshev polynomials

    CoordType get_dsi() const; // partial derivative of si with respect to rij

private:
    CoordType rcut_max;
    CoordType rcut_smooth;
};   // end of class CalcKsi


template <typename CoordType>
class Chebyshev1st { // Chebyshev polynomials of the first kind
public:
    Chebyshev1st();

    Chebyshev1st(int beta, CoordType rcut_max, CoordType rcut_smooth);

    Chebyshev1st(const Chebyshev1st& other);

    ~Chebyshev1st();

    void build(CoordType rij); // build Chebyshev polynomials

    const int get_beta() const; // get order of Chebyshev polynomials

    const CoordType* get_vals() const; // get Chebyshev polynomials

    const CoordType* get_ders() const; // get derivative of Chebyshev polynomials

    const CoordType* get_ders2r() const; // get partial derivative of Chebyshev polynomials with respect to rij

    void show() const; // show Chebyshev polynomials

private:
    int beta;
    CalcKsi<CoordType> si;
    CoordType rcut_max;
    CoordType rcut_smooth;
    CoordType* vals;    // Chebyshev polynomials
    CoordType* ders;    // partial derivative of Chebyshev polynomials with respect to si
    CoordType* ders2r;     // partial derivative of Chebyshev polynomials with respect to rij
};   // end of class Chebyshev1st


/*
below is the implementation of the template classes
*/

template <typename CoordType>
CalcKsi<CoordType>::CalcKsi() {}; // default constructor

/**
 * @brief Constructor for the CalcKsi class.
 * 
 * @param rcut_max The maximum cutoff radius.
 * @param rcut_smooth The smoothing cutoff radius.
 * @tparam CoordType The type of the coordinates.
 */
template <typename CoordType>
CalcKsi<CoordType>::CalcKsi(CoordType rcut_max, CoordType rcut_smooth) {
    this->rcut_max = rcut_max;
    this->rcut_smooth = rcut_smooth;
} // constructor

/**
 * @brief Copy constructor for the CalcKsi class.
 * 
 * @param other The CalcKsi object to copy.
*/
template <typename CoordType>
CalcKsi<CoordType>::CalcKsi(const CalcKsi& other) {
    this->rcut_max = other.rcut_max;
    this->rcut_smooth = other.rcut_smooth;
} // copy constructor

/**
 * @brief Get the value of si.
 * 
 * @param rij $r_{ij} = \sqrt{x_{ij}^2 + y_{ij}^2 + z_{ij}^2}$.  
 * The squared root of the sum of the squared displacement vectors between atoms i and j.
*/
template <typename CoordType>
CoordType CalcKsi<CoordType>::get_si(CoordType rij) const {
    return (2.0 * rij - (this->rcut_max + this->rcut_smooth)) / (this->rcut_max - this->rcut_smooth);
} // get_si
    
/**
 * @brief Get the derivative of si. 
 * partial derivative of si with respect to rij.
*/
template <typename CoordType>
CoordType CalcKsi<CoordType>::get_dsi() const {
    return 2.0 / (this->rcut_max - this->rcut_smooth);
} // get_dsi


template <typename CoordType>
Chebyshev1st<CoordType>::Chebyshev1st() {}; // default constructor

/**
 * @brief Constructor for the Chebyshev1st class.
 * 
 * @param beta The order of the Chebyshev polynomials.
 * @param rcut_max The maximum cutoff radius.
 * @param rcut_smooth The smoothing cutoff radius.
 * @tparam CoordType The type of the coordinates.
 */
template <typename CoordType>
Chebyshev1st<CoordType>::Chebyshev1st(int beta, CoordType rcut_max, CoordType rcut_smooth)
    : beta(beta), rcut_max(rcut_max), rcut_smooth(rcut_smooth), si(rcut_max, rcut_smooth) {
    this->vals = new CoordType[beta];
    std::fill_n(this->vals, beta, CoordType());
    this->ders = new CoordType[beta];
    std::fill_n(this->ders, beta, CoordType());
    this->ders2r = new CoordType[beta];
    std::fill_n(this->ders2r, beta, CoordType());
} // constructor

/**
 * @brief Copy constructor for the Chebyshev1st class.
 * 
 * @param other The Chebyshev1st object to copy.
*/
template <typename CoordType>
Chebyshev1st<CoordType>::Chebyshev1st(const Chebyshev1st& other)
    : beta(other.beta), rcut_max(other.rcut_max), rcut_smooth(other.rcut_smooth), si(other.si) {
    this->vals = new CoordType[beta];
    std::copy_n(other.vals, beta, this->vals);
    this->ders = new CoordType[beta];
    std::copy_n(other.ders, beta, this->ders);
    this->ders2r = new CoordType[beta];
    std::copy_n(other.ders2r, beta, this->ders2r);
} // copy constructor

/**
 * @brief Destructor for the Chebyshev1st class.
*/
template <typename CoordType>
Chebyshev1st<CoordType>::~Chebyshev1st() {
    delete[] this->vals;
    delete[] this->ders;
    delete[] this->ders2r;
} // destructor

/**
 * @brief Build the Chebyshev polynomials.
*/
template <typename CoordType>
void Chebyshev1st<CoordType>::build(CoordType rij) {
    CoordType si = this->si.get_si(rij);
    if ((rij >= this->rcut_smooth) && (rij <= this->rcut_max)) { // Chebyshv polynomials only defined on [-1, 1].
        this->vals[0] = 1.0;
        this->vals[1] = si;
        this->ders[0] = 0.0;
        this->ders[1] = 1.0;
        this->ders2r[0] = 0.0;
        this->ders2r[1] = 1.0 * this->si.get_dsi();
        for (int i = 2; i < this->beta; i++) {
            this->vals[i] = 2.0 * si * this->vals[i - 1] - this->vals[i - 2];
            this->ders[i] = 2.0 * si * this->ders[i - 1] + 2.0 * this->vals[i - 1] - this->ders[i - 2];
            this->ders2r[i] = this->ders[i] * this->si.get_dsi(); // \frac{d T_n}{dr_{ij}} = \frac{d T_n}{d \kxi} \frac{d \kxi}{dr_{ij}}
        }
    } else {
        std::fill_n(this->vals, this->beta, CoordType());
        std::fill_n(this->ders, this->beta, CoordType());
        std::fill_n(this->ders2r, this->beta, CoordType());
    }
} // build

/**
 * @brief Get the order of the Chebyshev polynomials.
*/
template <typename CoordType>
const int Chebyshev1st<CoordType>::get_beta() const {
    return (const int)this->beta;
} // get_beta

/**
 * @brief Get the Chebyshev polynomials.
*/
template <typename CoordType>
const CoordType* Chebyshev1st<CoordType>::get_vals() const {
    return (const CoordType*)this->vals;
} // get_vals

/**
 * @brief Get the derivative of the Chebyshev polynomials.
*/
template <typename CoordType>
const CoordType* Chebyshev1st<CoordType>::get_ders() const {
    return (const CoordType*)this->ders;
} // get_ders

/**
 * @brief Get the partial derivative of the Chebyshev polynomials with respect to rij.
*/
template <typename CoordType>
const CoordType* Chebyshev1st<CoordType>::get_ders2r() const {
    return (const CoordType*)this->ders2r;
} // get_ders2r

/**
 * @brief Show the Chebyshev polynomials.
*/
template <typename CoordType>
void Chebyshev1st<CoordType>::show() const {
    std::cout << "Chebyshev polynomials of the first kind:" << std::endl;
    std::cout << "beta = " << this->beta << std::endl;
    std::cout << "rcut_max = " << this->rcut_max << std::endl;
    std::cout << "rcut_smooth = " << this->rcut_smooth << std::endl;
    std::cout << "vals = ";
    for (int i = 0; i < this->beta; i++) {
        std::cout << this->vals[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "ders = ";
    for (int i = 0; i < this->beta; i++) {
        std::cout << this->ders[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "ders2r = ";
    for (int i = 0; i < this->beta; i++) {
        std::cout << this->ders2r[i] << " ";
    }
    std::cout << std::endl;
} // show


#endif // BASIC_H