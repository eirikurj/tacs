/*
  This file is part of TACS: The Toolkit for the Analysis of Composite
  Structures, a parallel finite-element code for structural and
  multidisciplinary design optimization.

  Copyright (C) 2014 Georgia Tech Research Corporation

  TACS is licensed under the Apache License, Version 2.0 (the
  "License"); you may not use this software except in compliance with
  the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef TACS_ELEMENT_ALGEBRA_H
#define TACS_ELEMENT_ALGEBRA_H

/*
  A header file with lots of useful linear algebra. Note that this is
  designed to be included in .c/.cpp files directly.
*/

#include "TACSObject.h"

/*
  Compute the cross-product

  out = a*(x cross y)

  input:
  a:    the scalar multiplier
  x:    the first input 3-vector
  y:    the second input 3-vector

  output:
  out:  the resulting vector
*/
static inline void crossProduct( const TacsScalar a,
                                 const TacsScalar x[],
                                 const TacsScalar y[],
                                 TacsScalar out[] ){
  out[0] = a*(x[1]*y[2] - x[2]*y[1]);
  out[1] = a*(x[2]*y[0] - x[0]*y[2]);
  out[2] = a*(x[0]*y[1] - x[1]*y[0]);
}

/*
  Compute the cross-product and add it to the output vector

  out = a*(x cross y)

  input:
  a:    the scalar multiplier
  x:    the first input 3-vector
  y:    the second input 3-vector

  output:
  out:  the resulting vector
*/
static inline void crossProductAdd( const TacsScalar a,
                                    const TacsScalar x[],
                                    const TacsScalar y[],
                                    TacsScalar out[] ){
  out[0] += a*(x[1]*y[2] - x[2]*y[1]);
  out[1] += a*(x[2]*y[0] - x[0]*y[2]);
  out[2] += a*(x[0]*y[1] - x[1]*y[0]);
}

/*
  Scale the vector by the given scalar

  input
  a:   the scalar
  x:   the vector
*/
static inline void vec3Scale( const TacsScalar a,
                              TacsScalar x[] ){
  x[0] *= a;
  x[1] *= a;
  x[2] *= a;
}

/*
  Compute the dot-product of two vectors

  return = x^{T}*y

  input:
  x:   the first vector
  y:   the second vector

  returns: the dot product
*/
static inline TacsScalar vec3Dot( const TacsScalar x[],
                                  const TacsScalar y[] ){
  return (x[0]*y[0] + x[1]*y[1] + x[2]*y[2]);
}

/*
  Add the product using an AXPY operation

  y <- y + alpha*x

  input:
  a:    the alpha scalar
  x:    the input vector

  in/out:
  y:    the result
*/
static inline void vec3Axpy( const TacsScalar a,
                             const TacsScalar x[],
                             TacsScalar y[] ){
  y[0] += a*x[0];
  y[1] += a*x[1];
  y[2] += a*x[2];
}



/*
  Scale the vector by the given scalar

  input
  a:   the scalar
  x:   the vector
*/
static inline void vec2Scale( const TacsScalar a,
                              TacsScalar x[] ){
  x[0] *= a;
  x[1] *= a;
}

/*
  Compute the dot-product of two vectors

  return = x^{T}*y

  input:
  x:   the first vector
  y:   the second vector

  returns: the dot product
*/
static inline TacsScalar vec2Dot( const TacsScalar x[],
                                  const TacsScalar y[] ){
  return (x[0]*y[0] + x[1]*y[1]);
}

/*
  Add the product using an AXPY operation

  y <- y + alpha*x

  input:
  a:    the alpha scalar
  x:    the input vector

  in/out:
  y:    the result
*/
static inline void vec2Axpy( const TacsScalar a,
                             const TacsScalar x[],
                             TacsScalar y[] ){
  y[0] += a*x[0];
  y[1] += a*x[1];
}

/*
  Compute the outer product of two vectors

  C <- a*b^{T}

  input:
  a:   the first vector
  b:   the second vector

  output:
  C:   the resulting matrix
*/
static inline void vec3x3Outer( const TacsReal a[],
                                const TacsReal b[],
                                TacsReal C[] ){
  C[0] = a[0]*b[0];
  C[1] = a[0]*b[1];
  C[2] = a[0]*b[2];

  C[3] = a[1]*b[0];
  C[4] = a[1]*b[1];
  C[5] = a[1]*b[2];

  C[6] = a[2]*b[0];
  C[7] = a[2]*b[1];
  C[8] = a[2]*b[2];
}

/*
  Add the outer product of two vectors

  C <- a*b^{T}

  input:
  a:   the first vector
  b:   the second vector

  output:
  C:   the resulting matrix
*/
static inline void vec3x3OuterAdd( const TacsScalar alpha,
                                   const TacsScalar a[],
                                   const TacsScalar b[],
                                   TacsScalar C[] ){
  C[0] += alpha*a[0]*b[0];
  C[1] += alpha*a[0]*b[1];
  C[2] += alpha*a[0]*b[2];

  C[3] += alpha*a[1]*b[0];
  C[4] += alpha*a[1]*b[1];
  C[5] += alpha*a[1]*b[2];

  C[6] += alpha*a[2]*b[0];
  C[7] += alpha*a[2]*b[1];
  C[8] += alpha*a[2]*b[2];
}

/*
  Compute the outer product of two vectors

  C <- a*b^{T}

  input:
  a:   the first vector
  b:   the second vector

  output:
  C:   the resulting matrix
*/
static inline void vec3x3Outer( const TacsComplex a[],
                                const TacsReal b[],
                                TacsComplex C[] ){
  C[0] = a[0]*b[0];
  C[1] = a[0]*b[1];
  C[2] = a[0]*b[2];

  C[3] = a[1]*b[0];
  C[4] = a[1]*b[1];
  C[5] = a[1]*b[2];

  C[6] = a[2]*b[0];
  C[7] = a[2]*b[1];
  C[8] = a[2]*b[2];
}

/*
  Add the outer product of two vectors

  C <- a*b^{T}

  input:
  a:   the first vector
  b:   the second vector

  output:
  C:   the resulting matrix
*/
static inline void vec3x3OuterAdd( const TacsComplex alpha,
                                   const TacsComplex a[],
                                   const TacsReal b[],
                                   TacsComplex C[] ){
  C[0] += alpha*a[0]*b[0];
  C[1] += alpha*a[0]*b[1];
  C[2] += alpha*a[0]*b[2];

  C[3] += alpha*a[1]*b[0];
  C[4] += alpha*a[1]*b[1];
  C[5] += alpha*a[1]*b[2];

  C[6] += alpha*a[2]*b[0];
  C[7] += alpha*a[2]*b[1];
  C[8] += alpha*a[2]*b[2];
}

/*
  Compute the outer product of two vectors

  C <- a*b^{T}

  input:
  a:   the first vector
  b:   the second vector

  output:
  C:   the resulting matrix
*/
static inline void vec2x2Outer( const TacsScalar a[],
                                const TacsScalar b[],
                                TacsScalar C[] ){
  C[0] = a[0]*b[0];
  C[1] = a[0]*b[1];

  C[2] = a[1]*b[0];
  C[3] = a[1]*b[1];
}

/*
  Add the outer product of two vectors

  C <- a*b^{T}

  input:
  a:   the first vector
  b:   the second vector

  output:
  C:   the resulting matrix
*/
static inline void vec2x2OuterAdd( const TacsScalar alpha,
                                   const TacsScalar a[],
                                   const TacsScalar b[],
                                   TacsScalar C[] ){
  C[0] += alpha*a[0]*b[0];
  C[1] += alpha*a[0]*b[1];

  C[2] += alpha*a[1]*b[0];
  C[3] += alpha*a[1]*b[1];
}

/*
  Compute the outer product of two vectors

  C <- a*b^{T}

  input:
  a:   the first vector
  b:   the second vector

  output:
  C:   the resulting matrix
*/
static inline void vec2x2Outer( const TacsComplex a[],
                                const TacsReal b[],
                                TacsComplex C[] ){
  C[0] = a[0]*b[0];
  C[1] = a[0]*b[1];

  C[2] = a[1]*b[0];
  C[3] = a[1]*b[1];
}

/*
  Add the outer product of two vectors

  C <- a*b^{T}

  input:
  a:   the first vector
  b:   the second vector

  output:
  C:   the resulting matrix
*/
static inline void vec2x2OuterAdd( const TacsComplex alpha,
                                   const TacsComplex a[],
                                   const TacsReal b[],
                                   TacsComplex C[] ){
  C[0] += alpha*a[0]*b[0];
  C[1] += alpha*a[0]*b[1];

  C[2] += alpha*a[1]*b[0];
  C[3] += alpha*a[1]*b[1];
}

/*
  Compute the derivative of x/||x||_{2} w.r.t. x using x and the norm
  of x.

  This code computes a 3x3 matrix that takes the form:

  d(x/||x||_{2})/dx = (I*||x||^2 + x*x^{T})/||x||^3
*/
static inline void vec3NormDeriv( TacsScalar nrm,
                                  const TacsScalar x[],
                                  TacsScalar D[] ){
  TacsScalar s = 1.0/(nrm*nrm*nrm);
  TacsScalar t = nrm*nrm;

  D[0] = s*(t - x[0]*x[0]);
  D[1] =-s*x[0]*x[1];
  D[2] =-s*x[0]*x[2];

  D[3] =-s*x[1]*x[0];
  D[4] = s*(t - x[1]*x[1]);
  D[5] =-s*x[1]*x[2];

  D[6] =-s*x[2]*x[0];
  D[7] =-s*x[2]*x[1];
  D[8] = s*(t - x[2]*x[2]);
}

/*
  Compute y <- A*x

  input:
  A:   the 3x3 input matrix in row-major order
  x:   the input 3-vector

  output:
  y:   the resulting vector
*/
static inline void mat3x3Mult( const TacsScalar A[],
                               const TacsScalar x[],
                               TacsScalar y[] ){
  y[0] = A[0]*x[0] + A[1]*x[1] + A[2]*x[2];
  y[1] = A[3]*x[0] + A[4]*x[1] + A[5]*x[2];
  y[2] = A[6]*x[0] + A[7]*x[1] + A[8]*x[2];
}

/*
  Compute y <- A*x

  input:
  A:   the 3x3 input matrix in row-major order
  x:   the input 3-vector

  output:
  y:   the resulting vector
*/
static inline void mat3x3Mult( const TacsComplex A[],
                               const TacsReal x[],
                               TacsComplex y[] ){
  y[0] = A[0]*x[0] + A[1]*x[1] + A[2]*x[2];
  y[1] = A[3]*x[0] + A[4]*x[1] + A[5]*x[2];
  y[2] = A[6]*x[0] + A[7]*x[1] + A[8]*x[2];
}

/*
  Compute y <- A*x

  input:
  A:   the 2x2 input matrix in row-major order
  x:   the input 2-vector

  output:
  y:   the resulting vector
*/
static inline void mat2x2Mult( const TacsScalar A[],
                               const TacsScalar x[],
                               TacsScalar y[] ){
  y[0] = A[0]*x[0] + A[1]*x[1];
  y[1] = A[2]*x[0] + A[3]*x[1];
}

/*
  Compute y <- A*x

  input:
  A:   the 2x2 input matrix in row-major order
  x:   the input 2-vector

  output:
  y:   the resulting vector
*/
static inline void mat2x2Mult( const TacsComplex A[],
                               const TacsReal x[],
                               TacsComplex y[] ){
  y[0] = A[0]*x[0] + A[1]*x[1];
  y[1] = A[2]*x[0] + A[3]*x[1];
}

/*
  Compute y <- A^{T}*x

  input:
  A:   the 3x3 input matrix in row-major order
  x:   the input 3-vector

  output:
  y:   the resulting vector
*/
static inline void mat3x3MultTrans( const TacsScalar A[],
                                    const TacsScalar x[],
                                    TacsScalar y[] ){
  y[0] = A[0]*x[0] + A[3]*x[1] + A[6]*x[2];
  y[1] = A[1]*x[0] + A[4]*x[1] + A[7]*x[2];
  y[2] = A[2]*x[0] + A[5]*x[1] + A[8]*x[2];
}

/*
  Compute y <- A^{T}*x

  input:
  A:   the 3x3 input matrix in row-major order
  x:   the input 3-vector

  output:
  y:   the resulting vector
*/
static inline void mat3x3MultTrans( const TacsComplex A[],
                                    const TacsReal x[],
                                    TacsComplex y[] ){
  y[0] = A[0]*x[0] + A[3]*x[1] + A[6]*x[2];
  y[1] = A[1]*x[0] + A[4]*x[1] + A[7]*x[2];
  y[2] = A[2]*x[0] + A[5]*x[1] + A[8]*x[2];
}

/*
  Compute y <- A^{T}*x

  input:
  A:   the 2x2 input matrix in row-major order
  x:   the input 2-vector

  output:
  y:   the resulting vector
*/
static inline void mat2x2MultTrans( const TacsScalar A[],
                                    const TacsScalar x[],
                                    TacsScalar y[] ){
  y[0] = A[0]*x[0] + A[2]*x[1];
  y[1] = A[1]*x[0] + A[3]*x[1];
}

/*
  Compute y <- A^{T}*x

  input:
  A:   the 2x2 input matrix in row-major order
  x:   the input 2-vector

  output:
  y:   the resulting vector
*/
static inline void mat2x2MultTrans( const TacsComplex A[],
                                    const TacsReal x[],
                                    TacsComplex y[] ){
  y[0] = A[0]*x[0] + A[2]*x[1];
  y[1] = A[1]*x[0] + A[3]*x[1];
}

/*
  Compute y <- y + A*x

  input:
  A:   the 3x3 input matrix in row-major order
  x:   the input 3-vector

  in/out:
  y:   the resulting vector
*/
static inline void mat3x3MultAdd( const TacsScalar A[],
                                  const TacsScalar x[],
                                  TacsScalar y[] ){
  y[0] += A[0]*x[0] + A[1]*x[1] + A[2]*x[2];
  y[1] += A[3]*x[0] + A[4]*x[1] + A[5]*x[2];
  y[2] += A[6]*x[0] + A[7]*x[1] + A[8]*x[2];
}

/*
  Compute y <- y + A*x

  input:
  A:   the 2x2 input matrix in row-major order
  x:   the input 2-vector

  in/out:
  y:   the resulting vector
*/
static inline void mat2x2MultAdd( const TacsScalar A[],
                                  const TacsScalar x[],
                                  TacsScalar y[] ){
  y[0] += A[0]*x[0] + A[1]*x[1];
  y[1] += A[2]*x[0] + A[3]*x[1];
}

/*
  Compute y <- y + A*x

  input:
  A:   the 3x3 input matrix in row-major order
  x:   the input 3-vector

  in/out:
  y:   the resulting vector
*/
static inline void mat3x3MultTransAdd( const TacsScalar A[],
                                       const TacsScalar x[],
                                       TacsScalar y[] ){
  y[0] += A[0]*x[0] + A[3]*x[1] + A[6]*x[2];
  y[1] += A[1]*x[0] + A[4]*x[1] + A[7]*x[2];
  y[2] += A[2]*x[0] + A[5]*x[1] + A[8]*x[2];
}

/*
  Compute y <- y + A*x

  input:
  A:   the 2x2 input matrix in row-major order
  x:   the input 2-vector

  in/out:
  y:   the resulting vector
*/
static inline void mat2x2MultTransAdd( const TacsScalar A[],
                                       const TacsScalar x[],
                                       TacsScalar y[] ){
  y[0] += A[0]*x[0] + A[2]*x[1];
  y[1] += A[1]*x[0] + A[3]*x[1];
}

/*
  Compute the inner product with a 3x3 matrix:

  return:  x^{T}*A*y

  input:
  A:   a 3x3 matrix in row-major order
  x:   a 3-vector
  y:   a 3-vector
*/
static inline TacsScalar mat3x3Inner( const TacsScalar A[],
                                      const TacsScalar x[],
                                      const TacsScalar y[] ){
  return (x[0]*(A[0]*y[0] + A[1]*y[1] + A[2]*y[2]) +
          x[1]*(A[3]*y[0] + A[4]*y[1] + A[5]*y[2]) +
          x[2]*(A[6]*y[0] + A[7]*y[1] + A[8]*y[2]));
}

/*
  Compute the inner product with a 2x2 matrix:

  return:  x^{T}*A*y

  input:
  A:   a 2x2 matrix in row-major order
  x:   a 2-vector
  y:   a 2-vector
*/
static inline TacsScalar mat2x2Inner( const TacsScalar A[],
                                      const TacsScalar x[],
                                      const TacsScalar y[] ){
  return (x[0]*(A[0]*y[0] + A[1]*y[1]) +
          x[1]*(A[2]*y[0] + A[3]*y[1]));
}

/*
  Compute y^{T}*A*x

  input:
  A:   the 3x3 input matrix in row-major order
  x:   the input 3-vector
  y:   the resulting vector

  returns:  the inner product
*/
static inline TacsScalar mat3x3SymmInner( const TacsScalar A[],
                                          const TacsScalar x[],
                                          const TacsScalar y[] ){
  return (y[0]*(A[0]*x[0] + A[1]*x[1] + A[2]*x[2]) +
          y[1]*(A[1]*x[0] + A[3]*x[1] + A[4]*x[2]) +
          y[2]*(A[2]*x[0] + A[4]*x[1] + A[5]*x[2]));
}

/*
  Compute y^{T}*A*x

  input:
  A:   the 2x2 input matrix in row-major order
  x:   the input 2-vector
  y:   the resulting vector

  returns:  the inner product
*/
static inline TacsScalar mat2x2SymmInner( const TacsScalar A[],
                                          const TacsScalar x[],
                                          const TacsScalar y[] ){
  return (y[0]*(A[0]*x[0] + A[1]*x[1]) +
          y[1]*(A[1]*x[0] + A[2]*x[1]));
}

/*
  Compute y <- y + A*x

  input:
  A:   the 3x3 input matrix in row-major order
  x:   the input 3-vector

  in/out:
  y:   the resulting vector
*/
static inline void mat3x3SymmMult( const TacsScalar A[],
                                   const TacsScalar x[],
                                   TacsScalar y[] ){
  y[0] = A[0]*x[0] + A[1]*x[1] + A[2]*x[2];
  y[1] = A[1]*x[0] + A[3]*x[1] + A[4]*x[2];
  y[2] = A[2]*x[0] + A[4]*x[1] + A[5]*x[2];
}

/*
  Compute y <- y + A*x

  input:
  A:   the 2x2 input matrix in row-major order
  x:   the input 2-vector

  in/out:
  y:   the resulting vector
*/
static inline void mat2x2SymmMult( const TacsScalar A[],
                                   const TacsScalar x[],
                                   TacsScalar y[] ){
  y[0] = A[0]*x[0] + A[1]*x[1];
  y[1] = A[1]*x[0] + A[2]*x[1];
}

/*
  Compute y <- y + A*x

  input:
  A:   the 3x3 input matrix in row-major order
  x:   the input 3-vector

  in/out:
  y:   the resulting vector
*/
static inline void mat3x3SymmMultAdd( const TacsScalar A[],
                                      const TacsScalar x[],
                                      TacsScalar y[] ){
  y[0] += A[0]*x[0] + A[1]*x[1] + A[2]*x[2];
  y[1] += A[1]*x[0] + A[3]*x[1] + A[4]*x[2];
  y[2] += A[2]*x[0] + A[4]*x[1] + A[5]*x[2];
}

/*
  Compute y <- y + A*x

  input:
  A:   the 2x2 input matrix in row-major order
  x:   the input 2-vector

  in/out:
  y:   the resulting vector
*/
static inline void mat2x2SymmMultAdd( const TacsScalar A[],
                                      const TacsScalar x[],
                                      TacsScalar y[] ){
  y[0] += A[0]*x[0] + A[1]*x[1];
  y[1] += A[1]*x[0] + A[2]*x[1];
}

/*
  Compute C = A*B

  input:
  A:   the first 3x3 input matrix in row-major order
  B:   the second 3x3 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat3x3MatMult( const TacsScalar A[],
                                  const TacsScalar B[],
                                  TacsScalar C[] ){
  C[0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6];
  C[3] = A[3]*B[0] + A[4]*B[3] + A[5]*B[6];
  C[6] = A[6]*B[0] + A[7]*B[3] + A[8]*B[6];

  C[1] = A[0]*B[1] + A[1]*B[4] + A[2]*B[7];
  C[4] = A[3]*B[1] + A[4]*B[4] + A[5]*B[7];
  C[7] = A[6]*B[1] + A[7]*B[4] + A[8]*B[7];

  C[2] = A[0]*B[2] + A[1]*B[5] + A[2]*B[8];
  C[5] = A[3]*B[2] + A[4]*B[5] + A[5]*B[8];
  C[8] = A[6]*B[2] + A[7]*B[5] + A[8]*B[8];
}

/*
  Compute C = A*B

  input:
  A:   the first 2x2 input matrix in row-major order
  B:   the second 2x2 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat2x2MatMult( const TacsScalar A[],
                                  const TacsScalar B[],
                                  TacsScalar C[] ){
  C[0] = A[0]*B[0] + A[1]*B[2];
  C[2] = A[2]*B[0] + A[3]*B[2];

  C[1] = A[0]*B[1] + A[1]*B[3];
  C[3] = A[2]*B[1] + A[3]*B[3];
}

/*
  Compute C = A*B^{T}

  input:
  A:   the first 3x3 input matrix in row-major order
  B:   the second 3x3 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat3x3MatTransMult( const TacsScalar A[],
                                       const TacsScalar B[],
                                       TacsScalar C[] ){
  C[0] = A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
  C[3] = A[3]*B[0] + A[4]*B[1] + A[5]*B[2];
  C[6] = A[6]*B[0] + A[7]*B[1] + A[8]*B[2];

  C[1] = A[0]*B[3] + A[1]*B[4] + A[2]*B[5];
  C[4] = A[3]*B[3] + A[4]*B[4] + A[5]*B[5];
  C[7] = A[6]*B[3] + A[7]*B[4] + A[8]*B[5];

  C[2] = A[0]*B[6] + A[1]*B[7] + A[2]*B[8];
  C[5] = A[3]*B[6] + A[4]*B[7] + A[5]*B[8];
  C[8] = A[6]*B[6] + A[7]*B[7] + A[8]*B[8];
}

/*
  Compute C = A*B^{T}

  input:
  A:   the first 2x2 input matrix in row-major order
  B:   the second 2x2 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat2x2MatTransMult( const TacsScalar A[],
                                       const TacsScalar B[],
                                       TacsScalar C[] ){
  C[0] = A[0]*B[0] + A[1]*B[1];
  C[1] = A[0]*B[2] + A[1]*B[3];

  C[2] = A[2]*B[0] + A[3]*B[1];
  C[3] = A[2]*B[2] + A[3]*B[3];
}

/*
  Compute C = A^{T}*B

  input:
  A:   the first 3x3 input matrix in row-major order
  B:   the second 3x3 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat3x3TransMatMult( const TacsScalar A[],
                                       const TacsScalar B[],
                                       TacsScalar C[] ){
  C[0] = A[0]*B[0] + A[3]*B[3] + A[6]*B[6];
  C[1] = A[0]*B[1] + A[3]*B[4] + A[6]*B[7];
  C[2] = A[0]*B[2] + A[3]*B[5] + A[6]*B[8];

  C[3] = A[1]*B[0] + A[4]*B[3] + A[7]*B[6];
  C[4] = A[1]*B[1] + A[4]*B[4] + A[7]*B[7];
  C[5] = A[1]*B[2] + A[4]*B[5] + A[7]*B[8];

  C[6] = A[2]*B[0] + A[5]*B[3] + A[8]*B[6];
  C[7] = A[2]*B[1] + A[5]*B[4] + A[8]*B[7];
  C[8] = A[2]*B[2] + A[5]*B[5] + A[8]*B[8];
}

/*
  Compute C = A^{T}*B

  input:
  A:   the first 2x2 input matrix in row-major order
  B:   the second 2x2 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat2x2TransMatMult( const TacsScalar A[],
                                       const TacsScalar B[],
                                       TacsScalar C[] ){
  C[0] = A[0]*B[0] + A[2]*B[2];
  C[1] = A[0]*B[1] + A[2]*B[3];

  C[2] = A[1]*B[0] + A[3]*B[2];
  C[3] = A[1]*B[1] + A[3]*B[3];
}

/*
  Compute C += A*B

  input:
  A:   the first 3x3 input matrix in row-major order
  B:   the second 3x3 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat3x3MatMultAdd( const TacsScalar A[],
                                     const TacsScalar B[],
                                     TacsScalar C[] ){
  C[0] += A[0]*B[0] + A[1]*B[3] + A[2]*B[6];
  C[3] += A[3]*B[0] + A[4]*B[3] + A[5]*B[6];
  C[6] += A[6]*B[0] + A[7]*B[3] + A[8]*B[6];

  C[1] += A[0]*B[1] + A[1]*B[4] + A[2]*B[7];
  C[4] += A[3]*B[1] + A[4]*B[4] + A[5]*B[7];
  C[7] += A[6]*B[1] + A[7]*B[4] + A[8]*B[7];

  C[2] += A[0]*B[2] + A[1]*B[5] + A[2]*B[8];
  C[5] += A[3]*B[2] + A[4]*B[5] + A[5]*B[8];
  C[8] += A[6]*B[2] + A[7]*B[5] + A[8]*B[8];
}

/*
  Compute C =+ A*B

  input:
  A:   the first 2x2 input matrix in row-major order
  B:   the second 2x2 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat2x2MatMultAdd( const TacsScalar A[],
                                     const TacsScalar B[],
                                     TacsScalar C[] ){
  C[0] += A[0]*B[0] + A[1]*B[2];
  C[2] += A[2]*B[0] + A[3]*B[2];

  C[1] += A[0]*B[1] + A[1]*B[3];
  C[3] += A[2]*B[1] + A[3]*B[3];
}

/*
  Compute C += A^{T}*B

  input:
  A:   the first 3x3 input matrix in row-major order
  B:   the second 3x3 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat3x3TransMatMultAdd( const TacsScalar A[],
                                          const TacsScalar B[],
                                          TacsScalar C[] ){
  C[0] += A[0]*B[0] + A[3]*B[3] + A[6]*B[6];
  C[1] += A[0]*B[1] + A[3]*B[4] + A[6]*B[7];
  C[2] += A[0]*B[2] + A[3]*B[5] + A[6]*B[8];

  C[3] += A[1]*B[0] + A[4]*B[3] + A[7]*B[6];
  C[4] += A[1]*B[1] + A[4]*B[4] + A[7]*B[7];
  C[5] += A[1]*B[2] + A[4]*B[5] + A[7]*B[8];

  C[6] += A[2]*B[0] + A[5]*B[3] + A[8]*B[6];
  C[7] += A[2]*B[1] + A[5]*B[4] + A[8]*B[7];
  C[8] += A[2]*B[2] + A[5]*B[5] + A[8]*B[8];
}

/*
  Compute C += A^{T}*B

  input:
  A:   the first 2x2 input matrix in row-major order
  B:   the second 2x2 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat2x2TransMatMultAdd( const TacsScalar A[],
                                          const TacsScalar B[],
                                          TacsScalar C[] ){
  C[0] += A[0]*B[0] + A[2]*B[2];
  C[2] += A[1]*B[0] + A[3]*B[2];

  C[1] += A[0]*B[1] + A[2]*B[3];
  C[3] += A[1]*B[1] + A[3]*B[3];
}

/*
  Compute C += A*B^{T}

  input:
  A:   the first 3x3 input matrix in row-major order
  B:   the second 3x3 input matrix in row-major order

  output:
  C:   the resulting matrix
*/
static inline void mat3x3MatTransMultAdd( const TacsScalar A[],
                                          const TacsScalar B[],
                                          TacsScalar C[] ){
  C[0] += A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
  C[3] += A[3]*B[0] + A[4]*B[1] + A[5]*B[2];
  C[6] += A[6]*B[0] + A[7]*B[1] + A[8]*B[2];

  C[1] += A[0]*B[3] + A[1]*B[4] + A[2]*B[5];
  C[4] += A[3]*B[3] + A[4]*B[4] + A[5]*B[5];
  C[7] += A[6]*B[3] + A[7]*B[4] + A[8]*B[5];

  C[2] += A[0]*B[6] + A[1]*B[7] + A[2]*B[8];
  C[5] += A[3]*B[6] + A[4]*B[7] + A[5]*B[8];
  C[8] += A[6]*B[6] + A[7]*B[7] + A[8]*B[8];
}

/*
  Multiply the matrix by

  [ A[0] A[1] A[2] ][ B[0]  B[1]  B[2]  B[3] ]
  [ A[3] A[4] A[5] ][ B[4]  B[5]  B[6]  B[7] ]
  [ A[6] A[7] A[8] ][ B[8]  B[9]  B[10] B[11] ]

  input:
  A:   a symmetric 3x3 matrix
  B:   a 3x4 matrix in row-major order

  output:
  C:   a 3x4 matrix in row-major order
*/
static inline void matMat3x4Mult( const TacsScalar A[],
                                  const TacsScalar B[],
                                  TacsScalar C[] ){
  C[0] = A[0]*B[0] + A[1]*B[4] + A[2]*B[8];
  C[1] = A[0]*B[1] + A[1]*B[5] + A[2]*B[9];
  C[2] = A[0]*B[2] + A[1]*B[6] + A[2]*B[10];
  C[3] = A[0]*B[3] + A[1]*B[7] + A[2]*B[11];

  C[4] = A[3]*B[0] + A[4]*B[4] + A[5]*B[8];
  C[5] = A[3]*B[1] + A[4]*B[5] + A[5]*B[9];
  C[6] = A[3]*B[2] + A[4]*B[6] + A[5]*B[10];
  C[7] = A[3]*B[3] + A[4]*B[7] + A[5]*B[11];

  C[8] = A[6]*B[0] + A[7]*B[4] + A[8]*B[8];
  C[9] = A[6]*B[1] + A[7]*B[5] + A[8]*B[9];
  C[10]= A[6]*B[2] + A[7]*B[6] + A[8]*B[10];
  C[11]= A[6]*B[3] + A[7]*B[7] + A[8]*B[11];
}

/*
  Multiply a symmetric matrix by a 3x4 row-major matrix

  [ A[0] A[1] A[2] ][ B[0]  B[1]  B[2]  B[3] ]
  [ A[1] A[3] A[4] ][ B[4]  B[5]  B[6]  B[7] ]
  [ A[2] A[4] A[5] ][ B[8]  B[9]  B[10] B[11] ]

  input:
  A:   a symmetric 3x3 matrix
  B:   a 3x4 matrix in row-major order

  output:
  C:   a 3x4 matrix in row-major order
*/
static inline void matSymmMat3x4Mult( const TacsScalar A[],
                                      const TacsScalar B[],
                                      TacsScalar C[] ){
  C[0] = A[0]*B[0] + A[1]*B[4] + A[2]*B[8];
  C[1] = A[0]*B[1] + A[1]*B[5] + A[2]*B[9];
  C[2] = A[0]*B[2] + A[1]*B[6] + A[2]*B[10];
  C[3] = A[0]*B[3] + A[1]*B[7] + A[2]*B[11];

  C[4] = A[1]*B[0] + A[3]*B[4] + A[4]*B[8];
  C[5] = A[1]*B[1] + A[3]*B[5] + A[4]*B[9];
  C[6] = A[1]*B[2] + A[3]*B[6] + A[4]*B[10];
  C[7] = A[1]*B[3] + A[3]*B[7] + A[4]*B[11];

  C[8] = A[2]*B[0] + A[4]*B[4] + A[5]*B[8];
  C[9] = A[2]*B[1] + A[4]*B[5] + A[5]*B[9];
  C[10]= A[2]*B[2] + A[4]*B[6] + A[5]*B[10];
  C[11]= A[2]*B[3] + A[4]*B[7] + A[5]*B[11];
}

/*
  Set the 3x3 matrix as a skew symmetric matrix

  C = a*b^{x}

  where a is a scalar and b is 3-vector

  input:
  a:    a scalar
  b:    a 3-vector

  output:
  C:    the result is added to this matrix
*/
static inline void setMatSkew( const TacsScalar a,
                               const TacsScalar b[],
                               TacsScalar C[] ){
  C[0] = 0.0;
  C[1] = -a*b[2];
  C[2] = a*b[1];

  C[3] = a*b[2];
  C[4] = 0.0;
  C[5] = -a*b[0];

  C[6] = -a*b[1];
  C[7] = a*b[0];
  C[8] = 0.0;
}

/*
  Add the skew matrix to a 3x3 matrix

  C += a*b^{x}

  where a is a scalar and b is 3-vector

  input:
  a:    a scalar
  b:    a 3-vector

  output:
  C:    the result is added to this matrix
*/
static inline void addMatSkew( const TacsScalar a,
                               const TacsScalar b[],
                               TacsScalar C[] ){
  C[1] -= a*b[2];
  C[2] += a*b[1];

  C[3] += a*b[2];
  C[5] -= a*b[0];

  C[6] -= a*b[1];
  C[7] += a*b[0];
}

/*
  Set the product of two skew matrices into the matrix

  C = a*b^{x}*c^{x} = a*(c*b^{T} - c^{T}*b*I)

  input:
  a:    a scalar
  b:    a 3-vector

  output:
  C:    the result is added to this matrix
*/
static inline void setMatSkewSkew( const TacsScalar a,
                                   const TacsScalar b[],
                                   const TacsScalar c[],
                                   TacsScalar C[] ){
  C[0] = -a*(c[1]*b[1] + c[2]*b[2]);
  C[1] = a*c[0]*b[1];
  C[2] = a*c[0]*b[2];

  C[3] = a*c[1]*b[0];
  C[4] = -a*(c[0]*b[0] + c[2]*b[2]);
  C[5] = a*c[1]*b[2];

  C[6] = a*c[2]*b[0];
  C[7] = a*c[2]*b[1];
  C[8] = -a*(c[0]*b[0] + c[1]*b[1]);
}

/*
  Add the product of two skew matrices to the matrix

  C += a*b^{x}*c^{x} = a*(c*b^{T} - c^{T}*b*I)

  input:
  a:    a scalar
  b:    a 3-vector

  output:
  C:    the result is added to this matrix
*/
static inline void addMatSkewSkew( const TacsScalar a,
                                   const TacsScalar b[],
                                   const TacsScalar c[],
                                   TacsScalar C[] ){
  C[0] -= a*(c[1]*b[1] + c[2]*b[2]);
  C[1] += a*c[0]*b[1];
  C[2] += a*c[0]*b[2];

  C[3] += a*c[1]*b[0];
  C[4] -= a*(c[0]*b[0] + c[2]*b[2]);
  C[5] += a*c[1]*b[2];

  C[6] += a*c[2]*b[0];
  C[7] += a*c[2]*b[1];
  C[8] -= a*(c[0]*b[0] + c[1]*b[1]);
}

/*
  Add a 3x3 block-matrix to a larger block matrix

  D[:,:] += a*A

  input:
  a:    the scalar
  A:    the 3x3 block matrix
  ldd:  the leading row-dimension of the D matrix

  in/out:
  D: the block matrix with in row-major order
*/
static inline void addBlockMat( const TacsScalar a,
                                const TacsScalar A[],
                                TacsScalar D[], const int ldd ){
  D[0] += a*A[0];
  D[1] += a*A[1];
  D[2] += a*A[2];

  D += ldd;
  D[0] += a*A[3];
  D[1] += a*A[4];
  D[2] += a*A[5];

  D += ldd;
  D[0] += a*A[6];
  D[1] += a*A[7];
  D[2] += a*A[8];
}

/*
  Add the transpose of a 3x3 block-matrix to a larger block matrix

  D[:,:] += a*A^{T}

  input:
  a:    the scalar
  A:    the 3x3 block matrix
  ldd:  the leading row-dimension of the D matrix

  in/out:
  D:    the block matrix with in row-major order
*/
static inline void addBlockMatTrans( const TacsScalar a,
                                     const TacsScalar A[],
                                     TacsScalar D[], const int ldd ){
  D[0] += a*A[0];
  D[1] += a*A[3];
  D[2] += a*A[6];

  D += ldd;
  D[0] += a*A[1];
  D[1] += a*A[4];
  D[2] += a*A[7];

  D += ldd;
  D[0] += a*A[2];
  D[1] += a*A[5];
  D[2] += a*A[8];
}

/*
  Add a vector to the matrix

  D[:,:] += a*A

  input:
  a:    the scalar
  A:    the 3-vector to add to the matrix
  ldd:  the leading row-dimension of the D matrix

  in/out:
  D:    the block matrix with in row-major order
*/
static inline void addVecMat( const TacsScalar a,
                              const TacsScalar A[],
                              TacsScalar D[],
                              const int ldd ){
  D[0] += a*A[0]; D += ldd;
  D[0] += a*A[1]; D += ldd;
  D[0] += a*A[2];
}

/*
  Add a 3x3 block-matrix to a larger block matrix

  D[:,:] += a*A

  input:
  a:    the scalar
  A:    the 3x3 block matrix
  ldd:  the leading row-dimension of the D matrix

  in/out:
  D: the block matrix with in row-major order
*/
static inline void addBlockSymmMat( const TacsScalar a,
                                    const TacsScalar A[],
                                    TacsScalar D[], const int ldd ){
  D[0] += a*A[0];
  D[1] += a*A[1];
  D[2] += a*A[2];

  D += ldd;
  D[0] += a*A[1];
  D[1] += a*A[3];
  D[2] += a*A[4];

  D += ldd;
  D[0] += a*A[2];
  D[1] += a*A[4];
  D[2] += a*A[5];
}

/*
  Add a scalar multiple of the identity matrix to the block matrix

  input:
  a:    the scalar
  ldd:  the leading row-dimension of the D matrix

  in/out:
  D:    the block matrix in row-major order
*/
static inline void addBlockIdent( const TacsScalar a,
                                  TacsScalar D[], const int ldd ){
  D[0] += a;

  D += ldd;
  D[1] += a;

  D += ldd;
  D[2] += a;
}

/*
  Add a 3x3 scalar skew-symmetric matrix to the D matrix

  input:
  a:    the scalar
  x:    the vector
  ldd:  the leading row-dimension of the D matrix

  in/out:
  D:    the block matrix in row-major order
*/
static inline void addBlockSkew( const TacsScalar a,
                                 const TacsScalar x[],
                                 TacsScalar D[], const int ldd ){
  D[1] -= a*x[2];
  D[2] += a*x[1];

  D += ldd;
  D[0] += a*x[2];
  D[2] -= a*x[0];

  D += ldd;
  D[0] -= a*x[1];
  D[1] += a*x[0];
}

/*
  Add the product of two skew-symm matrices to the D matrix

  D[:,:] += a*x^{x}*y^{x} = a*(y*x^{T} - I*x^{T}*y)

  input:
  a:    the scalar
  x:    the first vector
  y:    the second vectorr
  ldd:  the leading row-dimension of the D matrix

  in/out:
  D:    the block matrix in row-major order
*/
static inline void addBlockSkewSkew( const TacsScalar a,
                                     const TacsScalar x[],
                                     const TacsScalar y[],
                                     TacsScalar D[], const int ldd ){
  D[0] -= a*(x[1]*y[1] + x[2]*y[2]);
  D[1] += a*y[0]*x[1];
  D[2] += a*y[0]*x[2];

  D += ldd;
  D[0] += a*y[1]*x[0];
  D[1] -= a*(x[0]*y[0] + x[2]*y[2]);
  D[2] += a*y[1]*x[2];

  D += ldd;
  D[0] += a*y[2]*x[0];
  D[1] += a*y[2]*x[1];
  D[2] -= a*(x[0]*y[0] + x[1]*y[1]);
}

/*
  Compute the determinant of a 3x3 matrix

  input:
  A:        a 3x3 matrix in row-major order

  returns:  the determinant of A
*/
static inline TacsScalar det3x3( const TacsScalar A[] ){
  return (A[8]*(A[0]*A[4] - A[3]*A[1]) -
          A[7]*(A[0]*A[5] - A[3]*A[2]) +
          A[6]*(A[1]*A[5] - A[2]*A[4]));
}

/*
  Compute the derivative of the determinant with respect to the
  components of A
*/
static inline void det3x3Sens( const TacsScalar A[],
                               TacsScalar Ad[] ){
  Ad[0] = A[8]*A[4] - A[7]*A[5];
  Ad[1] = A[6]*A[5] - A[8]*A[3];
  Ad[2] = A[7]*A[3] - A[6]*A[4];

  Ad[3] = A[7]*A[2] - A[8]*A[1];
  Ad[4] = A[8]*A[0] - A[6]*A[2];
  Ad[5] = A[6]*A[1] - A[7]*A[0];

  Ad[6] = A[1]*A[5] - A[2]*A[4];
  Ad[7] = A[3]*A[2] - A[0]*A[5];
  Ad[8] = A[0]*A[4] - A[3]*A[1];
}

/*
  Compute the inverse of a 3x3 matrix

  input:
  A:          a 3x3 matrix in row major order

  output:
  Ainv:       the inverse of the 3x3 matrix

  returns:    the determinant of A
*/
static inline TacsScalar inv3x3( const TacsScalar A[],
                                 TacsScalar Ainv[] ){
  TacsScalar det = (A[8]*(A[0]*A[4] - A[3]*A[1]) -
                    A[7]*(A[0]*A[5] - A[3]*A[2]) +
                    A[6]*(A[1]*A[5] - A[2]*A[4]));
  TacsScalar detinv = 1.0/det;

  Ainv[0] = (A[4]*A[8] - A[5]*A[7])*detinv;
  Ainv[1] =-(A[1]*A[8] - A[2]*A[7])*detinv;
  Ainv[2] = (A[1]*A[5] - A[2]*A[4])*detinv;

  Ainv[3] =-(A[3]*A[8] - A[5]*A[6])*detinv;
  Ainv[4] = (A[0]*A[8] - A[2]*A[6])*detinv;
  Ainv[5] =-(A[0]*A[5] - A[2]*A[3])*detinv;

  Ainv[6] = (A[3]*A[7] - A[4]*A[6])*detinv;
  Ainv[7] =-(A[0]*A[7] - A[1]*A[6])*detinv;
  Ainv[8] = (A[0]*A[4] - A[1]*A[3])*detinv;

  return det;
}

/*
  Compute the sensitivity of the 3x3 inverse matrix

  input:
  Ainv:   The 3x3 inverse of the matrix
  Ainvd:  The derivative of the 3x3 inverse

  output:
  Ainvd:      derivative of the inverse of the 3x3 matrix
*/
static inline void inv3x3Sens( const TacsScalar Ainv[],
                               const TacsScalar Ainvd[],
                               TacsScalar Ad[] ){
  // d(Ainv_{kl})/d(A_{ij})
  //  = -Ainv_{kn}*delta_{ni}*delta{mj}*Ainv_{ml}
  //  = -Ainv_{ki}*Ainv_{jl}

  // Ad_{ij}
  //  = d(Ainv_{kl})/d(A_{ij})*Ainvd_{kl}
  //  = -Ainv_{ki}*Ainv_{jl}*Ainvd_{kl}

  // Ad = -Ainv^{T}*Ainvd*Ainv^{T}
  TacsScalar t[9];
  mat3x3TransMatMult(Ainv, Ainvd, t);
  mat3x3MatTransMult(t, Ainv, Ad);

  Ad[0] = -Ad[0];
  Ad[1] = -Ad[1];
  Ad[2] = -Ad[2];
  Ad[3] = -Ad[3];
  Ad[4] = -Ad[4];
  Ad[5] = -Ad[5];
  Ad[6] = -Ad[6];
  Ad[7] = -Ad[7];
  Ad[8] = -Ad[8];
}

/*
  Compute the determinant of a 3x3 matrix

  input:
  A:        a 3x3 matrix in row-major order

  returns:  the determinant of A
*/
static inline TacsScalar det2x2( const TacsScalar A[] ){
  return (A[0]*A[3] - A[1]*A[2]);
}

/*
  Compute the derivative of the determinant with respect to the
  components of A
*/
static inline void det2x2Sens( const TacsScalar A[],
                               TacsScalar Ad[] ){
  Ad[0] = A[3];
  Ad[1] = -A[2];
  Ad[2] = -A[1];
  Ad[3] = A[0];
}

/*
  Compute the inverse of a 3x3 matrix

  input:
  A:          a 3x3 matrix in row major order

  output:
  Ainv:       the inverse of the 3x3 matrix

  returns:    the determinant of A
*/
static inline TacsScalar inv2x2( const TacsScalar A[],
                                 TacsScalar Ainv[] ){
  TacsScalar det = A[0]*A[3] - A[1]*A[2];
  TacsScalar detinv = 1.0/det;

  Ainv[0] =   A[3]*detinv;
  Ainv[1] = - A[1]*detinv;

  Ainv[2] = - A[2]*detinv;
  Ainv[3] =   A[0]*detinv;

  return det;
}

/*
  Compute the sensitivity of the 3x3 inverse matrix

  input:
  Ainv:   The 3x3 inverse of the matrix
  Ainvd:  The derivative of the 3x3 inverse

  output:
  Ainvd:      derivative of the inverse of the 3x3 matrix
*/
static inline void inv2x2Sens( const TacsScalar Ainv[],
                               const TacsScalar Ainvd[],
                               TacsScalar Ad[] ){
  // d(Ainv_{kl})/d(A_{ij})
  //  = -Ainv_{kn}*delta_{ni}*delta{mj}*Ainv_{ml}
  //  = -Ainv_{ki}*Ainv_{jl}

  // Ad_{ij}
  //  = d(Ainv_{kl})/d(A_{ij})*Ainvd_{kl}
  //  = -Ainv_{ki}*Ainv_{jl}*Ainvd_{kl}

  // Ad = -Ainv^{T}*Ainvd*Ainv^{T}
  TacsScalar t[4];
  mat2x2TransMatMult(Ainv, Ainvd, t);
  mat2x2MatTransMult(t, Ainv, Ad);

  Ad[0] = -Ad[0];
  Ad[1] = -Ad[1];
  Ad[2] = -Ad[2];
  Ad[3] = -Ad[3];
}

#endif // TACS_ELEMENT_ALGEBRA_H
