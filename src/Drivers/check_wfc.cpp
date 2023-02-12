////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file check_wfc.cpp
 * @brief Miniapp to check individual wave function component against its
 * reference.
 */

#include <Utilities/Configuration.h>
#include <Particle/ParticleSet.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Particle/DistanceTable.h>
#include <Numerics/Containers.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/RandomGenerator.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/Jastrow/PolynomialFunctor3D.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctor.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrow.h>
#include <QMCWaveFunctions/SPOSet_builder.h>
#include <QMCWaveFunctions/DiracDeterminantRef.h>
#include <QMCWaveFunctions/DiracDeterminant.h>
#include <Utilities/qmcpack_version.h>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;

void print_help()
{
  cout << "usage:";
  cout << '\n' << "  check_wfc [-hvV] [-f wfc_component] [-g \"n0 n1 n2\"]";
  cout << '\n' << "            [-r rmax]";
  cout << '\n' << "options:";
  cout << '\n' << "  -f  specify wavefunction component to check";
  cout << '\n' << "      one of: J1, J2, J3, Det.       default: J2";
  cout << '\n' << "  -g  set the 3D tiling.             default: 1 1 1";
  cout << '\n' << "  -h  print help and exit";
  cout << '\n' << "  -r  set the Rmax.                  default: 1.7";
  cout << '\n' << "  -v  verbose output";
  cout << '\n' << "  -V  print version information and exit";
  app_summary() << std::endl;

  exit(1); // print help and exit
}

template<typename T>
T check_grads(TinyVector<T, 3>& grad, TinyVector<T, 3>& grad_ref, bool use_relative_error)
{
  if (use_relative_error)
    return sqrt(dot(grad - grad_ref, grad - grad_ref) / dot(grad, grad_ref));
  else
    return sqrt(dot(grad - grad_ref, grad - grad_ref));
}

template<typename T>
T check_val(T val, T val_ref, bool use_relative_error)
{
  if (use_relative_error)
    return abs((val - val_ref) / val_ref);
  else
    return abs(val - val_ref);
}

int main(int argc, char** argv)
{
  // clang-format off
  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::ParticlePos    ParticlePos;
  typedef ParticleSet::PosType          PosType;
  // clang-format on

  // use the global generator

  int na = 1;
  int nb = 1;
  int nc = 1;
  RealType Rmax(1.7);
  string wfc_name("J2");

  bool use_offload = true;
  bool verbose     = false;

  int opt;
  while (optind < argc)
  {
    if ((opt = getopt(argc, argv, "huvVf:g:r:s:")) != -1)
    {
      switch (opt)
      {
      case 'f': // Wave function component
        wfc_name = optarg;
        break;
      case 'g': // tiling1 tiling2 tiling3
        sscanf(optarg, "%d %d %d", &na, &nb, &nc);
        break;
      case 'h':
        print_help();
        break;
      case 'r': // rmax
        Rmax = atof(optarg);
        break;
      case 'u':
        use_offload = false;
        break;
      case 'v':
        verbose = true;
        break;
      case 'V':
        print_version(true);
        return 1;
      default:
        print_help();
      }
    }
    else // disallow non-option arguments
    {
      cerr << "Non-option arguments not allowed" << endl;
      print_help();
    }
  }

  print_version(verbose);

  if (verbose)
    outputManager.setVerbosity(Verbosity::HIGH);
  else
    outputManager.setVerbosity(Verbosity::LOW);

  if (wfc_name != "J1" && wfc_name != "J2" && wfc_name != "J3" && wfc_name != "JeeI" && wfc_name != "Det")
  {
    cerr << "Uknown wave funciton component:  " << wfc_name << endl << endl;
    print_help();
  }

  Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);

  // setup ions
  std::unique_ptr<ParticleSet> ions_ptr = build_ions(tmat, use_offload);
  auto& ions(*ions_ptr);

  // list of accumulated errors
  double evaluateLog_v_err = 0.0;
  double evaluateLog_g_err = 0.0;
  double evaluateLog_l_err = 0.0;
  double evalGrad_g_err    = 0.0;
  double ratioGrad_r_err   = 0.0;
  double ratioGrad_g_err   = 0.0;
  double evaluateGL_g_err  = 0.0;
  double evaluateGL_l_err  = 0.0;
  double ratio_err         = 0.0;

  PrimeNumberSet<uint32_t> myPrimes;

  std::unique_ptr<SPOSet> spo_main;
  {
    RandomGenerator<RealType> random_th(myPrimes[0]);
    const int norb = count_electrons(*ions_ptr, 1) / 2;
    if (wfc_name == "Det")
      spo_main = build_SPOSet(false, 40, 40, 40, norb, 1, ions_ptr->getSimulationCell().getPrimLattice().R);
  }

// clang-format off
  #pragma omp parallel reduction(+:evaluateLog_v_err,evaluateLog_g_err,evaluateLog_l_err,evalGrad_g_err) \
   reduction(+:ratioGrad_r_err,ratioGrad_g_err,evaluateGL_g_err,evaluateGL_l_err,ratio_err)
  // clang-format on
  {
    int ip = omp_get_thread_num();

    // create generator within the thread
    RandomGenerator<RealType> random_th(myPrimes[ip]);

    auto els_ptr = build_els(ions, random_th, use_offload);
    auto& els(*els_ptr);

    const int nions = ions.getTotalNum();
    const int nels  = els.getTotalNum();
    const int nels3 = 3 * nels;

    ParticleSet els_ref(els);

    // create tables
    els.addTable(els);
    els_ref.addTable(els_ref);
    const int ei_TableID = els_ref.addTable(ions);

    ParticlePos delta(nels);

    vector<RealType> ur(nels);
    random_th.generate_uniform(ur.data(), nels);

    std::unique_ptr<WaveFunctionComponent> wfc;
    std::unique_ptr<WaveFunctionComponent> wfc_ref;

    bool use_relative_error(false);

    if (wfc_name == "J2")
    {
      TwoBodyJastrow<BsplineFunctor<RealType>>* J = new TwoBodyJastrow<BsplineFunctor<RealType>>(els);
      buildJ2(*J, els.getLattice().WignerSeitzRadius);
      wfc.reset(dynamic_cast<WaveFunctionComponentPtr>(J));
      cout << "Built J2" << endl;
      miniqmcreference::TwoBodyJastrowRef<BsplineFunctor<RealType>>* J_ref =
          new miniqmcreference::TwoBodyJastrowRef<BsplineFunctor<RealType>>(els_ref);
      buildJ2(*J_ref, els.getLattice().WignerSeitzRadius);
      wfc_ref.reset(dynamic_cast<WaveFunctionComponentPtr>(J_ref));
      cout << "Built J2_ref" << endl;
    }
    else if (wfc_name == "J1")
    {
      OneBodyJastrow<BsplineFunctor<RealType>>* J = new OneBodyJastrow<BsplineFunctor<RealType>>(ions, els);
      buildJ1(*J, els.getLattice().WignerSeitzRadius);
      wfc.reset(dynamic_cast<WaveFunctionComponentPtr>(J));
      cout << "Built J1" << endl;
      miniqmcreference::OneBodyJastrowRef<BsplineFunctor<RealType>>* J_ref =
          new miniqmcreference::OneBodyJastrowRef<BsplineFunctor<RealType>>(ions, els_ref);
      buildJ1(*J_ref, els.getLattice().WignerSeitzRadius);
      wfc_ref.reset(dynamic_cast<WaveFunctionComponentPtr>(J_ref));
      cout << "Built J1_ref" << endl;
    }
    else if (wfc_name == "JeeI" || wfc_name == "J3")
    {
      ThreeBodyJastrow<PolynomialFunctor3D>* J = new ThreeBodyJastrow<PolynomialFunctor3D>(ions, els);
      buildJeeI(*J, els.getLattice().WignerSeitzRadius);
      wfc.reset(dynamic_cast<WaveFunctionComponentPtr>(J));
      cout << "Built JeeI" << endl;
      miniqmcreference::ThreeBodyJastrowRef<PolynomialFunctor3D>* J_ref =
          new miniqmcreference::ThreeBodyJastrowRef<PolynomialFunctor3D>(ions, els_ref);
      buildJeeI(*J_ref, els.getLattice().WignerSeitzRadius);
      wfc_ref.reset(dynamic_cast<WaveFunctionComponentPtr>(J_ref));
      cout << "Built JeeI_ref" << endl;
    }
    else if (wfc_name == "Det")
    {
      auto spo  = build_SPOSet_view(false, *spo_main, 1, 0);
      auto* Det = new DiracDeterminant<>(std::move(spo), 0, 31);
      wfc.reset(dynamic_cast<WaveFunctionComponentPtr>(Det));
      cout << "Built Det" << endl;
      auto spo_ref  = build_SPOSet_view(false, *spo_main, 1, 0);
      auto* Det_ref = new miniqmcreference::DiracDeterminantRef<>(std::move(spo_ref), 0, 31);
      wfc_ref.reset(dynamic_cast<WaveFunctionComponentPtr>(Det_ref));
      cout << "Built Det_ref" << endl;
      use_relative_error = true;
    }

    constexpr RealType czero(0);

    // compute distance tables
    els.update();
    els_ref.update();

    {
      els.G = czero;
      els.L = czero;
      wfc->evaluateLog(els, els.G, els.L);

      els_ref.G = czero;
      els_ref.L = czero;
      wfc_ref->evaluateLog(els_ref, els_ref.G, els_ref.L);

      cout << "Check values " << wfc->LogValue << " " << els.G[12] << " " << els.L[12] << endl;
      cout << "Check values ref " << wfc_ref->LogValue << " " << els_ref.G[12] << " " << els_ref.L[12] << endl << endl;
      cout << "evaluateLog::V Error = " << (wfc->LogValue - wfc_ref->LogValue) / nels << endl;
      evaluateLog_v_err += std::fabs((wfc->LogValue - wfc_ref->LogValue) / nels);
      {
        double g_err = 0.0;
        for (int iel = 0; iel < nels; ++iel)
        {
          RealType d = check_grads(els.G[iel], els_ref.G[iel], use_relative_error);
          g_err += d;
        }
        cout << "evaluateLog::G Error = " << g_err / nels << endl;
        evaluateLog_g_err += std::fabs(g_err / nels);
      }
      {
        double l_err = 0.0;
        for (int iel = 0; iel < nels; ++iel)
        {
          l_err += check_val(els.L[iel], els_ref.L[iel], use_relative_error);
        }
        cout << "evaluateLog::L Error = " << l_err / nels << endl;
        evaluateLog_l_err += std::fabs(l_err / nels);
      }

      random_th.generate_normal(&delta[0][0], nels3);
      double g_eval  = 0.0;
      double r_ratio = 0.0;
      double g_ratio = 0.0;

      int naccepted = 0;

      for (int iel = 0; iel < nels; ++iel)
      {
        PosType grad_soa = wfc->evalGrad(els, iel);

        PosType grad_ref = wfc_ref->evalGrad(els_ref, iel);
        g_eval += check_grads(grad_soa, grad_ref, use_relative_error);

        els.makeMove(iel, delta[iel]);
        els_ref.makeMove(iel, delta[iel]);

        grad_soa       = 0;
        RealType r_soa = wfc->ratioGrad(els, iel, grad_soa);
        grad_ref       = 0;
        RealType r_ref = wfc_ref->ratioGrad(els_ref, iel, grad_ref);

        g_ratio += check_grads(grad_soa, grad_ref, use_relative_error);
        r_ratio += abs(r_soa / r_ref - 1);

        if (ur[iel] < r_ref)
        {
          wfc->acceptMove(els, iel);
          els.acceptMove(iel);

          wfc_ref->acceptMove(els_ref, iel);
          els_ref.acceptMove(iel);
          naccepted++;
        }
        else
        {
          els.rejectMove(iel);
          els_ref.rejectMove(iel);
        }
      }

      wfc->completeUpdates();
      wfc_ref->completeUpdates();

      cout << "Accepted " << naccepted << "/" << nels << endl;
      cout << "evalGrad::G      Error = " << g_eval / nels << endl;
      cout << "ratioGrad::G     Error = " << g_ratio / nels << endl;
      cout << "ratioGrad::Ratio Error = " << r_ratio / nels << endl;
      evalGrad_g_err += std::fabs(g_eval / nels);
      ratioGrad_g_err += std::fabs(g_ratio / nels);
      ratioGrad_r_err += std::fabs(r_ratio / nels);

      // nothing to do with J2 but needs for general cases
      els.donePbyP();
      els_ref.donePbyP();

      els.G = czero;
      els.L = czero;
      wfc->evaluateGL(els, els.G, els.L);

      els_ref.G = czero;
      els_ref.L = czero;
      wfc_ref->evaluateGL(els_ref, els_ref.G, els_ref.L);

      {
        double g_err = 0.0;
        for (int iel = 0; iel < nels; ++iel)
        {
          g_err += check_grads(els.G[iel], els_ref.G[iel], use_relative_error);
        }
        cout << "evaluteGL::G Error = " << g_err / nels << endl;
        evaluateGL_g_err += std::fabs(g_err / nels);
      }
      {
        double l_err = 0.0;
        for (int iel = 0; iel < nels; ++iel)
        {
          l_err += check_val(els.L[iel], els_ref.L[iel], use_relative_error);
        }
        cout << "evaluteGL::L Error = " << l_err / nels << endl;
        evaluateGL_l_err += std::fabs(l_err / nels);
      }

      // now ratio only
      r_ratio              = 0.0;
      constexpr int nknots = 12;
      int nsphere          = 0;
      for (int jel = 0; jel < els_ref.getTotalNum(); ++jel)
      {
        const auto& dist = els_ref.getDistTableAB(ei_TableID).getDistRow(jel);
        for (int iat = 0; iat < nions; ++iat)
          if (dist[iat] < Rmax)
          {
            nsphere++;
            random_th.generate_uniform(&delta[0][0], nknots * 3);
            for (int k = 0; k < nknots; ++k)
            {
              els.makeMove(jel, delta[k]);
              RealType r_soa = wfc->ratio(els, jel);
              els.rejectMove(jel);

              els_ref.makeMove(jel, delta[k]);
              RealType r_ref = wfc_ref->ratio(els_ref, jel);
              els_ref.rejectMove(jel);
              r_ratio += abs(r_soa / r_ref - 1);
            }
          }
      }
      cout << "ratio with SphereMove  Error = " << r_ratio / nsphere << " # of moves =" << nsphere << endl;
      ratio_err += std::fabs(r_ratio / (nels * nknots));
    }
  } // end of omp parallel

  int np               = omp_get_max_threads();
  const RealType small = std::numeric_limits<RealType>::epsilon() * (wfc_name == "Det" ? 1e6 : 1e4);
  std::cout << "Passing Tolerance " << small << std::endl;
  bool fail = false;
  cout << std::endl;
  if (evaluateLog_v_err / np > small)
  {
    cout << "Fail in evaluateLog, V error =" << evaluateLog_v_err / np << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (evaluateLog_g_err / np > small)
  {
    cout << "Fail in evaluateLog, G error =" << evaluateLog_g_err / np << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (evaluateLog_l_err / np > small)
  {
    cout << "Fail in evaluateLog, L error =" << evaluateLog_l_err / np << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (evalGrad_g_err / np > small)
  {
    cout << "Fail in evalGrad, G error =" << evalGrad_g_err / np << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (ratioGrad_r_err / np > small)
  {
    cout << "Fail in ratioGrad, ratio error =" << ratioGrad_r_err / np << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (ratioGrad_g_err / np > small)
  {
    cout << "Fail in ratioGrad, G error =" << ratioGrad_g_err / np << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (evaluateGL_g_err / np > small)
  {
    cout << "Fail in evaluateGL, G error =" << evaluateGL_g_err / np << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (evaluateGL_l_err / np > small)
  {
    cout << "Fail in evaluateGL, L error =" << evaluateGL_l_err / np << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (ratio_err / np > small)
  {
    cout << "Fail in ratio, ratio error =" << ratio_err / np << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (!fail)
    cout << "All checks passed for " << wfc_name << std::endl;

  return 0;
}
