#ifndef TACS_COUPLED_THERMO_PLANE_STRESS_STIFFNESS_H
#define TACS_COUPLED_THERMO_PLANE_STRESS_STIFFNESS_H

/*
  Copyright (c) 2017 Graeme Kennedy. All rights reserved. 
  Not for commercial purposes.
*/
#include "PlaneStressStiffness.h"
/*
  This is the plane stress constitutive objects with thermal loading. 
*/
class CoupledThermoPlaneStressStiffness : public PlaneStressStiffness {
 public:
  CoupledThermoPlaneStressStiffness();
  CoupledThermoPlaneStressStiffness( TacsScalar _rho, TacsScalar E,
                                     TacsScalar nu, TacsScalar _alpha, 
                                     TacsScalar Tref, TacsScalar kcond );

  ~CoupledThermoPlaneStressStiffness();

  // Calculate the product B*u
  // ---------------------------------------------------------------
  virtual void calculateStress( const double pt[],
                                const TacsScalar strain[],
                                TacsScalar stress[] ) = 0;

  // Compute the design dependent thermal loading stress
  //---------------------------------------------------
  virtual TacsScalar getEffThermalAlpha( int vars_j ) = 0;
  virtual TacsScalar getReferenceTemperature() = 0;
  virtual void addThermalDVSens( const double pt[],
                                 const TacsScalar e[],
                                 TacsScalar alpha,
                                 const TacsScalar psi[],
                                 TacsScalar fdvSens[],
                                 int dvLen ) = 0;
  virtual void calculateConduction( const double pt[],
                                    const TacsScalar strain[],
                                    TacsScalar stress[] ) = 0;
  virtual void calculateThermal( const double pt[],
                                 const TacsScalar strain[],
                                 TacsScalar stress[] ) = 0;
  virtual void addConductionDVSens( const double pt[],
                                    const TacsScalar e[],
                                    TacsScalar alpha,
                                    const TacsScalar psi[],
                                    TacsScalar fdvSens[],
                                    int dvLen ) = 0;

  // Extra info about the constitutive class
  // ---------------------------------------
  const char *constitutiveName();
 protected:
  // The stiffness matrix
  TacsScalar Cmat[6], Tmat[3]; 
  TacsScalar rho, alpha,xw, Tref;
 private:
  static const char *constName;
};
#endif // TACS_THERMO_PLANE_STRESS_STIFFNESS_H
