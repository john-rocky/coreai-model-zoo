#include "D1BLAS.h"

#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

// Accelerate's cblas_ddot sums in an order that depends on where its operands start: below a 256-byte boundary
// the result differs from an aligned call in the last bits (round 3a: 288 of 400 random length-2048 dots at a
// 32-byte offset). NumPy's float64 arrays of these sizes start on a page (macOS malloc), so the operands are
// copied to page-aligned buffers first; cblas_dgemv gave NumPy's bits at every alignment measured, and is fed the
// same aligned copies.
static double *aligned_copy(const double *src, long count) {
    void *p = NULL;
    if (posix_memalign(&p, 16384, (size_t)count * sizeof(double)) != 0) {
        return NULL;
    }
    memcpy(p, src, (size_t)count * sizeof(double));
    return (double *)p;
}

int d1_matvec(const double *a, long m, long n, const double *x, double *y) {
    if (m <= 0) {
        return 0;
    }
    double *aa = aligned_copy(a, m * n);
    double *xa = aligned_copy(x, n);
    if (aa == NULL || xa == NULL) {
        free(aa);
        free(xa);
        return -1;
    }
    if (m == 1) {
        double sum = 0.0;   // DOUBLE_dot: a double accumulator from 0., one cblas_ddot chunk (n < NPY_CBLAS_CHUNK)
        sum += cblas_ddot(n, aa, 1, xa, 1);
        y[0] = sum;
    } else {
        // DOUBLE_gemv on a C-contiguous A: column-major, transposed, lda = n
        cblas_dgemv(CblasColMajor, CblasTrans, n, m, 1.0, aa, n, xa, 1, 0.0, y, 1);
    }
    free(aa);
    free(xa);
    return 0;
}
