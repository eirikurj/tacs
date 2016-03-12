#ifndef TACS_FE_MATRIX_H
#define TACS_FE_MATRIX_H

/*
  Parallel matrix based upon a finite-element partitioning of the problem

  Copyright (c) 2010 Graeme Kennedy. All rights reserved.
  Not for commercial purposes.
*/

#include "ScMat.h"

/*
  The finite-element Schur-based matrix implementation.

  This matrix class uses a substructuring approach to achieve decent
  parallel performance when the level of fill is high enough that the
  matrix factorization approaches a complete factorziation approach.

  Input:

  rmap: The variable map that defines the row-distribution of the
  global matrix. This is required for matrix-vector products.

  nlocal_vars, rowp, cols: The CSR non-zero structure of all local
  variables

  b_local_indices: The local indicies of the B-matrix
  b_map: The map from the global variables to the local indices

  c_local_indices: The local indices of the C-matrix
  c_map: The map from the global 
  
  bcs: The boundary conditions
*/
class FEMat : public ScMat {
 public:
  FEMat( TACSThreadInfo * thread_info, VarMap * _rmap, 
	 int nlocal_vars, const int * rowp, const int * cols, 
	 BVecIndices * b_local_indices, BVecDistribute * _b_map, 
         BVecIndices * c_local_indices, BVecDistribute * _c_map,
	 BCMap * _bcs = NULL );
  ~FEMat();
    
  // Functions for setting values in the matrix
  // ------------------------------------------
  void addValues( int nrow, const int * row, int ncol, const int * col,
                  int nv, int mv, const TacsScalar * values );  
  void addWeightValues( int nvars, const int *varp, const int *vars,
			const TacsScalar *weights,
			int nv, int mv, const TacsScalar *values );
  void applyBCs();
  TACSVec * createVec();  
 
 private:  
  BCMap * bcs; // Boundary conditions
  int Nb, Nc;  // Size of B/C
};

#endif