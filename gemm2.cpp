#include <intrin.h>
#include <malloc.h>
#include <memory.h>
#include "gemm2.h"

static __declspec(align(64)) int mask[32] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

// "kernel" 6x16
static void micro_6x16(	// completes 6x16 submatrix of C
	int K,		// cols in A = rows in B
	const float *A,	// 0-th row (from 6)
	int lda,	// stride of A (lda >= K for À and 1 for A^T)
	int step,	// 1 for À and lda for A^T
    const float *B,	// 0-th column (from 16)
	int ldb,	// stride of B (=16 if B is pre-packed)
	float *C,	// out -> 6õ16 rect in a big matrix
	int ldc,	// stride of C
	int cy,		// rows for output (<= 6)
	int cx		// cols for output (<= 16)
) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();	// 1st 6x8 half
    __m256 c01 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();	// 2nd 6x8 half
    __m256 b0, b1, a0, a1;			// sources to multiply
	//
	const int lda2 = 2*lda, lda3 = 3*lda, lda4 = 4*lda, lda5 = 5*lda;
    for (int k = 0; k < K; k++) {
        b0 = _mm256_loadu_ps(B + 0);	// b0 = {B[0],B[1],B[2],B[3],B[4],B[5],B[6],B[7]}
        b1 = _mm256_loadu_ps(B + 8);	// b1 = {B[8],...,B[15]}
        a0 = _mm256_set1_ps(A[0]);		// a0 = {*A,*A,*A,*A,*A,*A,*A,*A}
        a1 = _mm256_set1_ps(A[lda]);	// a1 = {A[lda] 8 times}
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        a0 = _mm256_set1_ps(A[lda2]);
        a1 = _mm256_set1_ps(A[lda3]);
        c20 = _mm256_fmadd_ps(a0, b0, c20);
        c21 = _mm256_fmadd_ps(a0, b1, c21);
        c30 = _mm256_fmadd_ps(a1, b0, c30);
        c31 = _mm256_fmadd_ps(a1, b1, c31);
        a0 = _mm256_set1_ps(A[lda4]);
        a1 = _mm256_set1_ps(A[lda5]);
        c40 = _mm256_fmadd_ps(a0, b0, c40);
        c41 = _mm256_fmadd_ps(a0, b1, c41);
        c50 = _mm256_fmadd_ps(a1, b0, c50);
        c51 = _mm256_fmadd_ps(a1, b1, c51);
        B += ldb; A += step;
    }
	// add and store result
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c00, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c01, _mm256_loadu_ps(C + 8)));
	if (cy < 2) return;
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c10, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c11, _mm256_loadu_ps(C + 8)));
	if (cy < 3) return;
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c20, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c21, _mm256_loadu_ps(C + 8)));
	if (cy < 4) return;
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c30, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c31, _mm256_loadu_ps(C + 8)));
	if (cy < 5) return;
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c40, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c41, _mm256_loadu_ps(C + 8)));
	if (cy < 6) return;
    C += ldc;
	if (cx < 16) {
		float tmp[16];
		_mm256_storeu_ps( tmp, c50 );
		_mm256_storeu_ps( tmp + 8, c51 );
		for (int k = 0; k < cx; k++) { C[k] += tmp[k]; }
	} else {
		_mm256_storeu_ps(C + 0, _mm256_add_ps(c50, _mm256_loadu_ps(C + 0)));
		_mm256_storeu_ps(C + 8, _mm256_add_ps(c51, _mm256_loadu_ps(C + 8)));
	}
}

static void zero_C( int M, int N, float *C, int ldc ) {
    for (int j, i = 0; i < M; ++i, C += ldc) {
        for (j = 0; j < (N - 7); j += 8)
            _mm256_storeu_ps( C + j, _mm256_setzero_ps() );
		while (j < N) { C[j++] = 0; }
	}
}

// D[16,K] := S[n,K] S not transponed (X first)
static void reorder_b( int K, const float *S, int ldb, float *D, int n ) {
	if (n < 16) {
		__m256i mx0 = _mm256_maskz_load_epi32( 0xff, &mask[16 - n] );
		__m256i mx8 = _mm256_maskz_load_epi32( 0xff, &mask[24 - n] );
		for (int k = K; --k >= 0; S += ldb, D += 16) {
			_mm256_storeu_ps( D, _mm256_maskload_ps(S, mx0) );	// 8 floats
			_mm256_storeu_ps( D + 8, _mm256_maskload_ps(S + 8, mx8) );	// +8 floats
		}
	} else {
		for (int k = K; --k >= 0; S += ldb, D += 16) {
			_mm256_storeu_ps( D + 0, _mm256_loadu_ps(S + 0) );	// 8 floats
			_mm256_storeu_ps( D + 8, _mm256_loadu_ps(S + 8) );	// +8 floats
		}
	}
}

// B[wb,cx] := A[cy,cx], A transponed (Y first)
static void transband( const float *A, int cx, int cy, float *B, int wb ) {
	if (cy < wb)	// zerofill in advance to prevent gaps in B
		memset( B, 0, cx*wb*sizeof(B[0]) );
	for ( ; --cy >= 0; B++) {
		float *b = B;
		for (int j = cx; --j >= 0; b += wb) { *b = *A++; }
	}
}

// B := A[1:K,1:n] = vert. stripe of A (K rows, n cols, n <= wb (wb=6))
static void transaT( const float *A, int K, int lda, float *B, int wb, int n ) {
	if (n < wb) {
		__m256i mx = _mm256_maskz_load_epi32( 0xff, &mask[16 - n] );
		for ( ; --K >= 0; B += wb, A += lda) {
			_mm256_storeu_ps( B, _mm256_maskload_ps(A, mx) );
		}
	} else
		for ( ; --K >= 0; B += wb, A += lda) {
			_mm256_storeu_ps( B, _mm256_loadu_ps(A) );
		}
}

static void gemm2j(		// C = A*B (single thread)
	int tab,	// see TAB_* bits
	int M,		// rows in Ñ = rows in À (or cols in A if A^T)
	int N,		// cols in Ñ = cols in Â (or rows in Â if B^T)
	int K,		// cols in À (or A^T) = rows in Â (or B^T)
	const float *A,
	int lda,	// stride of À if A^T (=M if !TAB_PARALLEL)
	const float *B, 
	int ldb,	// stride of B (>= N)
	float *C,
	int ldc		// stride of C (>= N)
) {
	float *buf = (float*)_mm_malloc( (16*K + 16)*sizeof(buf[0]), 64 );
	int i, j, ij, n, m;
	int di = tab & TAB_TRANSP_A? 1 : K, da = tab & TAB_TRANSP_A? 6 : 1;
	//
    for (j = 0; j < N; j += 16) {
		n = _min(16, N - j);
		if (tab & TAB_TRANSP_B)	// B^T
	        transband( B + j*K, K, n, buf, 16 );
		else
			reorder_b( K, B + j, ldb, buf, n );
        for (i = 0, ij = j; i < M; i += 6, ij += 6*ldc) {
			m = _min(6, M - i);
			if (!(tab & TAB_ZEROED_C))
				zero_C( m, n, C + ij, ldc );	// C_ij = C[i:+6, j:j+16] := 0
//			micro_6x16( K, A + i*K, di, da, buf, 16, C + ij, ldc, m, n );	// C_ij += A[i, :]*B[:, j:+16]
			micro_6x16( K, A + i*K, 1, 6, buf, 16, C + ij, ldc, m, n );	// C_ij += A[i, :]*B[:, j:+16]
        }
    }
	_mm_free( buf );
}

#define __MT
#ifdef __MT
#include <windows.h>
#include "troika.h"

static int rob_gemm( void *host, int j ) {
	JOB_GEMM *jg = (JOB_GEMM*)host;
	int p = t_nya(), M = jg->M, N = jg->N, K = jg->K, T = jg->tab;
#if 0
	int m = (((M - 1)/6)/p + 1)*6;
	int i1 = j*m, i2 = __min( M, i1 + m ), di = T & 1? 1 : K;
	if (i2 > i1)
		gemm2j( T, i2 - i1, N, K, jg->A + i1*di, M, jg->B, N, jg->C + i1*N, N );
#else
	int n = (((N - 1)/16)/p + 1)*16;	// part of N for 1 thread
	int i1 = j*n, i2 = __min( N, i1 + n ), di = T & 2? K : 1;
	if (i2 > i1)
		gemm2j( T, M, i2 - i1, K, jg->A, M, jg->B + i1*di, N, jg->C + i1, N );
#endif
	return 0;
}
#endif

//////////////////////////////////////////////

void gemm2( int tab, int M, int N, int K, const float *A, const float *B, float *C ) {
	float *abu = (float*)_mm_malloc( (K*(M + 5))*sizeof(abu[0]), 64 );
	for (int i = 0; i < M; i += 6) {
		int m = _min(6, M - i);
		if (tab & TAB_TRANSP_A)
			transaT( A + i, K, M, abu + i*K, 6, m );
		else
			transband( A + i*K, K, m, abu + i*K, 6 );
	}
	if (tab & TAB_PARALLEL) {
		JOB_GEMM jg = {tab, M, N, K, abu, B, C};
		t_run( rob_gemm, &jg );
	} else
		gemm2j( tab, M, N, K, abu, M, B, N, C, N );
	_mm_free( abu );
}
