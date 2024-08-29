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
      inline int enzyme_dup;
      inline int enzyme_dupnoneed;
      inline int enzyme_out;
      inline int enzyme_const;
  
      using real_type = OptimizableFunctorBase::real_type;
      template<typename RT, typename... Args>
      RT __enzyme_autodiff(void*, Args...);
  
  //    using real_type = OptimizableFunctorBase::real_type;
  
      real_type functor_evaluate_wrapper(BsplineFunctor<real_type> functor, real_type r) {
      return functor.evaluate(r);
      }
  
      real_type functor_dudr_wrapper(BsplineFunctor<real_type> functor, real_type r) {
      return __enzyme_autodiff<real_type>((void*)functor_evaluate_wrapper, qmcad::enzyme_const, functor, qmcad::enzyme_out, r);
      }
  
      real_type functor_d2udr2_wrapper(BsplineFunctor<real_type> functor, real_type r) {
      return __enzyme_autodiff<real_type>((void*)functor_dudr_wrapper, qmcad::enzyme_const, functor, qmcad::enzyme_out, r);
      }
  }
  template<typename T>
  inline T BsplineFunctor<T>::evaluate(T r, T& dudr, T& d2udr2)
  {
    if (r >= cutoff_radius)
    {
      dudr = d2udr2 = 0.0;
      return 0.0;
    }
    dudr   = qmcad::functor_dudr_wrapper(*this, r);
    d2udr2   = qmcad::functor_d2udr2_wrapper(*this, r);
    return evaluate(r);
  }
  //BsplineFunctor<double>::evaluate(double r, double& dudr, double& d2udr2);
} // namespace qmcplusplus
