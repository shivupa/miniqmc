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

#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"

namespace qmcplusplus
{
  namespace qmcad{
 
      template<typename RT, typename... Args>
      RT __enzyme_autodiff(void*, Args...);
  
  //    using real_type = OptimizableFunctorBase::real_type;
      real_type BSpline_functor_evaluate_wrapper(const real_type& DeltaRInv, const real_type& cutoff_radius , const aligned_vector<real_type>& SplineCoefs, const TinyVector<real_type, 16>& A, const real_type& r) {
          if (r >= cutoff_radius)
            return 0.0;
          real_type r_timesRinv = r * DeltaRInv;
          real_type ipart, t;
          t     = std::modf(r_timesRinv, &ipart);
          int i = (int)ipart;
          real_type tp[4];
          tp[0] = t * t * t;
          tp[1] = t * t;
          tp[2] = t;
          tp[3] = 1.0;
          // clang-format off
          return
            (SplineCoefs[i+0]*(A[ 0]*tp[0] + A[ 1]*tp[1] + A[ 2]*tp[2] + A[ 3]*tp[3])+
             SplineCoefs[i+1]*(A[ 4]*tp[0] + A[ 5]*tp[1] + A[ 6]*tp[2] + A[ 7]*tp[3])+
             SplineCoefs[i+2]*(A[ 8]*tp[0] + A[ 9]*tp[1] + A[10]*tp[2] + A[11]*tp[3])+
             SplineCoefs[i+3]*(A[12]*tp[0] + A[13]*tp[1] + A[14]*tp[2] + A[15]*tp[3]));

       }
  
      real_type BSpline_functor_dudr_wrapper(const real_type& DeltaRInv, const real_type& cutoff_radius , const aligned_vector<real_type>& SplineCoefs, const TinyVector<real_type, 16>& A, const real_type& r) {
      return __enzyme_autodiff<real_type>((void*)BSpline_functor_evaluate_wrapper, qmcad::enzyme_const, DeltaRInv, qmcad::enzyme_const, cutoff_radius, qmcad::enzyme_const, SplineCoefs, qmcad::enzyme_const, A, qmcad::enzyme_out, r);
      }
  
      real_type BSpline_functor_d2udr2_wrapper(const real_type& DeltaRInv, const real_type& cutoff_radius , const aligned_vector<real_type>& SplineCoefs, const TinyVector<real_type, 16>& A, const real_type& r) {
      return __enzyme_autodiff<real_type>((void*)BSpline_functor_evaluate_wrapper, qmcad::enzyme_const, DeltaRInv, qmcad::enzyme_const, cutoff_radius, qmcad::enzyme_const, SplineCoefs, qmcad::enzyme_const, A, qmcad::enzyme_out, r);
      }
  }

  using real_type = OptimizableFunctorBase::real_type;

  template<typename T>
  real_type BsplineFunctor<T>::evaluate(real_type r, real_type& dudr, real_type& d2udr2)
  {
    if (r >= cutoff_radius)
    {
      dudr = d2udr2 = 0.0;
      return 0.0;
    }
    dudr   = qmcad::BSpline_functor_dudr_wrapper(DeltaRInv, cutoff_radius, SplineCoefs, A, r);
    d2udr2   = qmcad::BSpline_functor_d2udr2_wrapper(DeltaRInv, cutoff_radius, SplineCoefs, A, r);
    return qmcad::BSpline_functor_evaluate_wrapper(DeltaRInv, cutoff_radius, SplineCoefs, A, r);
    //return evaluate(r);
    // r *= DeltaRInv;
    // real_type ipart, t;
    // t     = std::modf(r, &ipart);
    // int i = (int)ipart;
    // real_type tp[4];
    // tp[0] = t * t * t;
    // tp[1] = t * t;
    // tp[2] = t;
    // tp[3] = 1.0;
    // // clang-format off
    // d2udr2 = DeltaRInv * DeltaRInv *
    //          (SplineCoefs[i+0]*(d2A[ 0]*tp[0] + d2A[ 1]*tp[1] + d2A[ 2]*tp[2] + d2A[ 3]*tp[3])+
    //           SplineCoefs[i+1]*(d2A[ 4]*tp[0] + d2A[ 5]*tp[1] + d2A[ 6]*tp[2] + d2A[ 7]*tp[3])+
    //           SplineCoefs[i+2]*(d2A[ 8]*tp[0] + d2A[ 9]*tp[1] + d2A[10]*tp[2] + d2A[11]*tp[3])+
    //           SplineCoefs[i+3]*(d2A[12]*tp[0] + d2A[13]*tp[1] + d2A[14]*tp[2] + d2A[15]*tp[3]));
    // dudr = DeltaRInv *
    //        (SplineCoefs[i+0]*(dA[ 0]*tp[0] + dA[ 1]*tp[1] + dA[ 2]*tp[2] + dA[ 3]*tp[3])+
    //         SplineCoefs[i+1]*(dA[ 4]*tp[0] + dA[ 5]*tp[1] + dA[ 6]*tp[2] + dA[ 7]*tp[3])+
    //         SplineCoefs[i+2]*(dA[ 8]*tp[0] + dA[ 9]*tp[1] + dA[10]*tp[2] + dA[11]*tp[3])+
    //         SplineCoefs[i+3]*(dA[12]*tp[0] + dA[13]*tp[1] + dA[14]*tp[2] + dA[15]*tp[3]));
    // return
    //   (SplineCoefs[i+0]*(A[ 0]*tp[0] + A[ 1]*tp[1] + A[ 2]*tp[2] + A[ 3]*tp[3])+
    //    SplineCoefs[i+1]*(A[ 4]*tp[0] + A[ 5]*tp[1] + A[ 6]*tp[2] + A[ 7]*tp[3])+
    //    SplineCoefs[i+2]*(A[ 8]*tp[0] + A[ 9]*tp[1] + A[10]*tp[2] + A[11]*tp[3])+
    //    SplineCoefs[i+3]*(A[12]*tp[0] + A[13]*tp[1] + A[14]*tp[2] + A[15]*tp[3]));
    // clang-format on
  }

  template struct BsplineFunctor<double >; 
  //BsplineFunctor<double>::evaluate(double r, double& dudr, double& d2udr2);
} // namespace qmcplusplus

