#ifndef __MATH_UTILS_HPP__
#define __MATH_UTILS_HPP__

//#include "LinearAlgebra.hpp"
#include "lineqn.h"

// Fast approximation for Gaussian: approximates exp(-x)
static inline float fast_negexp_approx(float x)
{
    return 1.0f / (1.0f + x + x*x);
}

template<int D, class REAL>
static inline void min_eigen_vec(
    REAL A[][D], REAL evec[], REAL* eval = NULL)
{
    REAL evals[D];
    eigdc<REAL,D>(A, evals);
    REAL emin = fabs(evals[0]);
    int imin = 0;
    for (int i = 1; i < D; i++) {
	if (emin > fabs(evals[i])) {
	    emin = fabs(evals[i]);
	    imin = i;
	}
    }
    for (int i = 0; i < D; i++)
	evec[i] = A[i][imin];
    if (eval) *eval = emin;
}

template<int D, class REAL>
static inline void max_eigen_vec(
    REAL A[][D], REAL evec[], REAL* eval = NULL)
{
    REAL evals[D];
    eigdc<REAL,D>(A, evals);
    REAL emax = fabs(evals[0]);
    int imax = 0;
    for (int i = 1; i < D; i++) {
	if (emax < fabs(evals[i])) {
	    emax = fabs(evals[i]);
	    imax = i;
	}
    }
    for (int i = 0; i < D; i++)
	evec[i] = A[i][imax];
    if (eval) *eval = emax;
}

#endif
