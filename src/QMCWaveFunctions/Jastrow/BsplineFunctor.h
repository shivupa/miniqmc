////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// John R. Gergely,  University of Illinois at Urbana-Champaign
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Miguel Morales, moralessilva2@llnl.gov,
//    Lawrence Livermore National Laboratory
// Raymond Clay III, j.k.rofling@gmail.com,
//    Lawrence Livermore National Laboratory
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jaron T. Krogel, krogeljt@ornl.gov,
//    Oak Ridge National Laboratory
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_BSPLINE_FUNCTOR_H
#define QMCPLUSPLUS_BSPLINE_FUNCTOR_H
#include "Numerics/OptimizableFunctorBase.h"
#include "CPU/SIMD/aligned_allocator.hpp"
#include <eigen3/Eigen/Core>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <cstdio>

/*!
 * @file BsplineFunctor.h
 */

namespace qmcplusplus
{
namespace qmcad
{

using namespace autodiff;
using Eigen::VectorXd;
using ad_real_type = var; 
using real_type = OptimizableFunctorBase::real_type;

inline ad_real_type BSpline_functor_evaluate_wrapper(const ad_real_type& DeltaRInv,
                                                     const ad_real_type& cutoff_radius,
                                                     const ArrayXvar& SplineCoefs,
                                                     const ArrayXvar& A,
                                                     const ad_real_type& r)
{
  ad_real_type u = 0.0;
  if (r >= cutoff_radius)
    return u;
  ad_real_type r_timesRinv = r * DeltaRInv;
  real_type ipart, t;
  t     = std::modf(val(r_timesRinv), &ipart);
  int i = (int)ipart;
  ad_real_type tp[4];
  tp[0] = t * t * t;
  tp[1] = t * t;
  tp[2] = t;
  tp[3] = 1.0;
  u     = (SplineCoefs[i + 0] * (A[0] * tp[0] + A[1] * tp[1] + A[2] * tp[2] + A[3] * tp[3]) +
       SplineCoefs[i + 1] * (A[4] * tp[0] + A[5] * tp[1] + A[6] * tp[2] + A[7] * tp[3]) +
       SplineCoefs[i + 2] * (A[8] * tp[0] + A[9] * tp[1] + A[10] * tp[2] + A[11] * tp[3]) +
       SplineCoefs[i + 3] * (A[12] * tp[0] + A[13] * tp[1] + A[14] * tp[2] + A[15] * tp[3]));
  return u;
};

inline void BSpline_functor_dudr_wrapper(const ad_real_type& DeltaRInv,
                                         const ad_real_type& cutoff_radius,
                                         const ArrayXvar& SplineCoefs,
                                         const ArrayXvar& A,
                                         const ad_real_type& r,
                                         ad_real_type& u,
                                         ad_real_type& out_du_dr)
{
  u            = BSpline_functor_evaluate_wrapper(DeltaRInv, cutoff_radius, SplineCoefs, A, r);
  auto [du_dr] = derivativesx(u, wrt(r)); // evaluate the first order derivatives of u
  out_du_dr    = du_dr;
};

inline void BSpline_functor_d2udr2_wrapper(const ad_real_type& DeltaRInv,
                                           const ad_real_type& cutoff_radius,
                                           const ArrayXvar& SplineCoefs,
                                           const ArrayXvar& A,
                                           const ad_real_type& r,
                                           ad_real_type& u,
                                           ad_real_type& out_du_dr,
                                           ad_real_type& out_d2u_dr2)
{
  u              = BSpline_functor_evaluate_wrapper(DeltaRInv, cutoff_radius, SplineCoefs, A, r);
  auto [du_dr]   = derivativesx(u, wrt(r)); // evaluate the first order derivatives of u
  auto [d2u_dr2] = derivativesx(du_dr, wrt(r));
  out_du_dr      = du_dr;
  out_d2u_dr2    = d2u_dr2;
};
} // namespace qmcad
} // namespace qmcplusplus

namespace qmcplusplus
{
template<class T>
struct BsplineFunctor : public OptimizableFunctorBase
{
  typedef real_type value_type;
  int NumParams;
  int Dummy;
  const TinyVector<real_type, 16> A, dA, d2A, d3A;
  aligned_vector<real_type> SplineCoefs;

  // static const real_type A[16], dA[16], d2A[16];
  real_type DeltaR, DeltaRInv;
  real_type CuspValue;
  real_type Y, dY, d2Y;
  // Stores the derivatives w.r.t. SplineCoefs
  // of the u, du/dr, and d2u/dr2
  std::vector<real_type> Parameters;
  std::vector<std::string> ParameterNames;
  std::string elementType, pairType;
  std::string fileName;

  int ResetCount;
  bool notOpt;
  bool periodic;

  /// constructor
  // clang-format off
  BsplineFunctor(real_type cusp=0.0) :
    NumParams(0),
    A(-1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
      3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
      -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
      1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0),
    dA(0.0, -0.5,  1.0, -0.5,
       0.0,  1.5, -2.0,  0.0,
       0.0, -1.5,  1.0,  0.5,
       0.0,  0.5,  0.0,  0.0),
    d2A(0.0, 0.0, -1.0,  1.0,
        0.0, 0.0,  3.0, -2.0,
        0.0, 0.0, -3.0,  1.0,
        0.0, 0.0,  1.0,  0.0),
    d3A(0.0, 0.0,  0.0, -1.0,
        0.0, 0.0,  0.0,  3.0,
        0.0, 0.0,  0.0, -3.0,
        0.0, 0.0,  0.0,  1.0),
    CuspValue(cusp), ResetCount(0), notOpt(false), periodic(true)
  {
    cutoff_radius = 0.0;
  }
  // clang-format on

  void resize(int n)
  {
    NumParams    = n;
    int numCoefs = NumParams + 4;
    int numKnots = numCoefs - 2;
    DeltaR       = cutoff_radius / (real_type)(numKnots - 1);
    DeltaRInv    = 1.0 / DeltaR;
    Parameters.resize(n);
    SplineCoefs.resize(numCoefs);
  }

  void reset()
  {
    int numCoefs = NumParams + 4;
    int numKnots = numCoefs - 2;
    DeltaR       = cutoff_radius / (real_type)(numKnots - 1);
    DeltaRInv    = 1.0 / DeltaR;
    for (int i = 0; i < SplineCoefs.size(); i++)
      SplineCoefs[i] = 0.0;
    // Ensure that cusp conditions is satsified at the origin
    SplineCoefs[1] = Parameters[0];
    SplineCoefs[2] = Parameters[1];
    SplineCoefs[0] = Parameters[1] - 2.0 * DeltaR * CuspValue;
    for (int i = 2; i < Parameters.size(); i++)
      SplineCoefs[i + 1] = Parameters[i];
  }

  void setupParameters(int n, real_type rcut, real_type cusp, std::vector<real_type>& params)
  {
    CuspValue     = cusp;
    cutoff_radius = rcut;
    resize(n);
    for (int i = 0; i < n; i++)
    {
      Parameters[i] = params[i];
    }
    reset();
  }

  /** compute value, gradient and laplacian for [iStart, iEnd) pairs
   * @param iStart starting particle index
   * @param iEnd ending particle index
   * @param _distArray distance arrUay
   * @param _valArray  u(r_j) for j=[iStart,iEnd)
   * @param _gradArray  du(r_j)/dr /r_j for j=[iStart,iEnd)
   * @param _lapArray  d2u(r_j)/dr2 for j=[iStart,iEnd)
   * @param distArrayCompressed temp storage to filter r_j < cutoff_radius
   * @param distIndices temp storage for the compressed index
   */
  // clang-format off
  void evaluateVGL(const int iat, const int iStart, const int iEnd, 
      const T* _distArray,  
      T* restrict _valArray,
      T* restrict _gradArray, 
      T* restrict _laplArray, 
      T* restrict distArrayCompressed, int* restrict distIndices ) const;
  // clang-format on

  /** evaluate sum of the pair potentials for [iStart,iEnd)
   * @param iStart starting particle index
   * @param iEnd ending particle index
   * @param _distArray distance arrUay
   * @param distArrayCompressed temp storage to filter r_j < cutoff_radius
   * @return \f$\sum u(r_j)\f$ for r_j < cutoff_radius
   */
  T evaluateV(const int iat,
              const int iStart,
              const int iEnd,
              const T* restrict _distArray,
              T* restrict distArrayCompressed) const;

  inline real_type evaluate(real_type r)
  {
    using namespace autodiff;
    using Eigen::VectorXd;

    VectorXvar ad_splinecoefs(SplineCoefs.size());
    for (auto val : SplineCoefs)
    {
      ad_splinecoefs << val;
    }
    VectorXvar ad_A(16);
    for (auto i = 0; i < 16; i++)
    {
      ad_A << A[i];
    }
    var ad_r             = r;
    var ad_DeltaRInv     = DeltaRInv;
    var ad_cutoff_radius = cutoff_radius;

    var ad_u = 0.0;

    ad_u = qmcad::BSpline_functor_evaluate_wrapper(ad_DeltaRInv, ad_cutoff_radius, ad_splinecoefs, ad_A, ad_r);

    real_type u = val(ad_u);
    return u;
  }

  inline real_type evaluate(real_type r, real_type& dudr, real_type& d2udr2)
  {
    using namespace autodiff;
    using Eigen::VectorXd;

    VectorXvar ad_splinecoefs(SplineCoefs.size());
    for (auto val : SplineCoefs)
    {
      ad_splinecoefs << val;
    }
    VectorXvar ad_A(16);
    for (auto i = 0; i < 16; i++)
    {
      ad_A << A[i];
    }
    var ad_r             = r;
    var ad_DeltaRInv     = DeltaRInv;
    var ad_cutoff_radius = cutoff_radius;

    var ad_u      = 0.0;
    var ad_dudr   = 0.0;
    var ad_d2udr2 = 0.0;

    qmcad::BSpline_functor_d2udr2_wrapper(ad_DeltaRInv, ad_cutoff_radius, ad_splinecoefs, ad_A, ad_r, ad_u, ad_dudr,
                                          ad_d2udr2);
    real_type u = val(ad_u);
    dudr        = val(ad_dudr);
    d2udr2      = val(ad_d2udr2);
    return u;
  }
};

template<typename T>
inline T BsplineFunctor<T>::evaluateV(const int iat,
                                      const int iStart,
                                      const int iEnd,
                                      const T* restrict _distArray,
                                      T* restrict distArrayCompressed) const
{
  const real_type* restrict distArray = _distArray + iStart;

  ASSUME_ALIGNED(distArrayCompressed);
  int iCount       = 0;
  const int iLimit = iEnd - iStart;

#pragma vector always
  for (int jat = 0; jat < iLimit; jat++)
  {
    real_type r = distArray[jat];
    // pick the distances smaller than the cutoff and avoid the reference atom
    if (r < cutoff_radius && iStart + jat != iat)
      distArrayCompressed[iCount++] = distArray[jat];
  }

  real_type d = 0.0;
  //#pragma omp simd reduction(+:d)
  for (int jat = 0; jat < iCount; jat++)
  {
    real_type r = distArrayCompressed[jat];
    r *= DeltaRInv;
    int i         = (int)r;
    real_type t   = r - real_type(i);
    real_type tp0 = t * t * t;
    real_type tp1 = t * t;
    real_type tp2 = t;

    real_type d1 = SplineCoefs[i + 0] * (A[0] * tp0 + A[1] * tp1 + A[2] * tp2 + A[3]);
    real_type d2 = SplineCoefs[i + 1] * (A[4] * tp0 + A[5] * tp1 + A[6] * tp2 + A[7]);
    real_type d3 = SplineCoefs[i + 2] * (A[8] * tp0 + A[9] * tp1 + A[10] * tp2 + A[11]);
    real_type d4 = SplineCoefs[i + 3] * (A[12] * tp0 + A[13] * tp1 + A[14] * tp2 + A[15]);
    d += (d1 + d2 + d3 + d4);
  }
  return d;
}

template<typename T>
inline void BsplineFunctor<T>::evaluateVGL(const int iat,
                                           const int iStart,
                                           const int iEnd,
                                           const T* _distArray,
                                           T* restrict _valArray,
                                           T* restrict _gradArray,
                                           T* restrict _laplArray,
                                           T* restrict distArrayCompressed,
                                           int* restrict distIndices) const
{
  real_type dSquareDeltaRinv = DeltaRInv * DeltaRInv;
  constexpr real_type cOne(1);

  //    START_MARK_FIRST();

  ASSUME_ALIGNED(distIndices);
  ASSUME_ALIGNED(distArrayCompressed);
  int iCount                 = 0;
  int iLimit                 = iEnd - iStart;
  const real_type* distArray = _distArray + iStart;
  real_type* valArray        = _valArray + iStart;
  real_type* gradArray       = _gradArray + iStart;
  real_type* laplArray       = _laplArray + iStart;

#pragma vector always
  for (int jat = 0; jat < iLimit; jat++)
  {
    real_type r = distArray[jat];
    if (r < cutoff_radius && iStart + jat != iat)
    {
      distIndices[iCount]         = jat;
      distArrayCompressed[iCount] = r;
      iCount++;
    }
  }

#pragma omp simd
  for (int j = 0; j < iCount; j++)
  {
    real_type r    = distArrayCompressed[j];
    int iScatter   = distIndices[j];
    real_type rinv = cOne / r;
    r *= DeltaRInv;
    int iGather   = (int)r;
    real_type t   = r - real_type(iGather);
    real_type tp0 = t * t * t;
    real_type tp1 = t * t;
    real_type tp2 = t;

    real_type sCoef0 = SplineCoefs[iGather + 0];
    real_type sCoef1 = SplineCoefs[iGather + 1];
    real_type sCoef2 = SplineCoefs[iGather + 2];
    real_type sCoef3 = SplineCoefs[iGather + 3];

    // clang-format off
    laplArray[iScatter] = dSquareDeltaRinv *
      (sCoef0*( d2A[ 2]*tp2 + d2A[ 3])+
       sCoef1*( d2A[ 6]*tp2 + d2A[ 7])+
       sCoef2*( d2A[10]*tp2 + d2A[11])+
       sCoef3*( d2A[14]*tp2 + d2A[15]));

    gradArray[iScatter] = DeltaRInv * rinv *
      (sCoef0*( dA[ 1]*tp1 + dA[ 2]*tp2 + dA[ 3])+
       sCoef1*( dA[ 5]*tp1 + dA[ 6]*tp2 + dA[ 7])+
       sCoef2*( dA[ 9]*tp1 + dA[10]*tp2 + dA[11])+
       sCoef3*( dA[13]*tp1 + dA[14]*tp2 + dA[15]));

    valArray[iScatter] = (sCoef0*(A[ 0]*tp0 + A[ 1]*tp1 + A[ 2]*tp2 + A[ 3])+
        sCoef1*(A[ 4]*tp0 + A[ 5]*tp1 + A[ 6]*tp2 + A[ 7])+
        sCoef2*(A[ 8]*tp0 + A[ 9]*tp1 + A[10]*tp2 + A[11])+
        sCoef3*(A[12]*tp0 + A[13]*tp1 + A[14]*tp2 + A[15]));
    // clang-format on
  }
}
} // namespace qmcplusplus
#endif
