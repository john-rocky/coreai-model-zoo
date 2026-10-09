// D1BLAS — the float64 products of the d1 readout through the very BLAS calls NumPy makes for `E @ h` when it is
// built against Accelerate (NumPy 2.x: numpy/_core/src/umath/matmul.c.src, `DOUBLE_matmul` -> `DOUBLE_gemv` /
// `DOUBLE_dot`), so the Swift host's logits equal host.py's `option_logits` bit for bit on the same machine:
//
//   m > 1   cblas_dgemv(CblasColMajor, CblasTrans, n, m, 1.0, A, n, x, 1, 0.0, y, 1)   (A row-major [m, n])
//   m == 1  y[0] = 0.0 + cblas_ddot(n, A, 1, x, 1)
//
// Accelerate's new interface with 64-bit integers (ACCELERATE_NEW_LAPACK + ACCELERATE_LAPACK_ILP64, the symbols NumPy
// 2.5.3 imports: cblas_dgemv$NEWLAPACK$ILP64, cblas_ddot$NEWLAPACK$ILP64); the classic CBLAS is deprecated in the
// SDK. A C target so that the two defines reach the Accelerate headers without unsafe Swift flags. The operands are
// copied to page-aligned buffers first, as NumPy's arrays start: cblas_ddot's summation order depends on alignment.

#ifndef D1BLAS_H
#define D1BLAS_H

/// y[m] = A[m, n] x[n] for a row-major (C-contiguous) A, as NumPy computes `A @ x` in float64. -> 0, or -1 when the
/// aligned copies could not be allocated (y untouched).
int d1_matvec(const double *a, long m, long n, const double *x, double *y);

#endif
