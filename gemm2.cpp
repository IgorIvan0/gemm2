#include <intrin.h>
#include <malloc.h>
#include <memory.h>
#include "gemm2.h"

static __declspec(align(64)) int mask[32] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

// "kernel" 6x16
static void micro_6x16(	// полностью заполняет фрагмент С (больше к нему не возвращаемся)
	int K,		// число точек в строке A = число строк в B
	const float *A,	// 0я строка (из 6)
	int lda,	// межстрочный интервал (lda >= K для А и 1 для A^T)
	int step,	// 1 для А и lda для A^T
    const float *B,	// колонка B шириной 16 (ре-упорядочено во временный буфер)
	int ldb,	// между 16-ками в В (=16 если В предварительно упаковали для скорости)
	float *C,	// куды скрадывать результат, поле 6х16 в большой матрице
	int ldc,	// между строками C
	int cy,		// строк во фрагменте (<= 6)
	int cx		// ширина фрагмента (<= 16)
) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();	// 6 регистров под 1ю 8ку
    __m256 c01 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();	// +6 под вторые 8
    __m256 b0, b1, a0, a1;			// +4 регистра под источники (их перемножаем)
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
	// 12 регистров -> + память (C)
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

// D[16,K] := S[n,K] S не повернуто (X вперед)
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

// B[wb,cx] := A[cy,cx], A повернуто (Y вперед)
static void transband( const float *A, int cx, int cy, float *B, int wb ) {
	if (cy < wb)	// строк меньше ширины ряда В, будут пропуски, занулить их заранее
		memset( B, 0, cx*wb*sizeof(B[0]) );
	for ( ; --cy >= 0; B++) {
		float *b = B;
		for (int j = cx; --j >= 0; b += wb) { *b = *A++; }
	}
}

// B := A[1:K,1:n] = полоса А шириной n <= wb (wb=6) из К строк
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

static void gemm2j(		// C = A*B
	int tab,	// 3 бита, 1: А повернута (A^T), 2: В^T, 4: занулить C
	int M,		// строк С = строк А (или столбов А если A^T)
	int N,		// столбов С = столбов В (или строк В если B^T)
	int K,		// столбов А (или A^T) = строк В (или B^T)
	const float *A,
	int lda,	// длина строки А для tab&1 (=M if !MT)
	const float *B, 
	int ldb,	// длина строки B (>= N)
	float *C,
	int ldc		// длина строки C (>= N)
) {
	float *buf = (float*)_mm_malloc( (16*K + 16)*sizeof(buf[0]), 64 );
	int i, j, ij, n, m;
	int di = tab & 1? 1 : K, da = tab & 1? 6 : 1;
	//
    for (j = 0; j < N; j += 16) {
		n = _min(16, N - j);
		if (tab & 2)	// B^T
	        transband( B + j*K, K, n, buf, 16 );
		else
			reorder_b( K, B + j, ldb, buf, n );
        for (i = 0, ij = j; i < M; i += 6, ij += 6*ldc) {
			m = _min(6, M - i);
			if (!(tab & 4))
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
	int m = (((M - 1)/6)/p + 1)*6;	// часть М для 1 нити
	int i1 = j*m, i2 = __min( M, i1 + m ), di = T & 1? 1 : K;
	if (i2 > i1)
		gemm2j( T, i2 - i1, N, K, jg->A + i1*di, M, jg->B, N, jg->C + i1*N, N );
#else
	int n = (((N - 1)/16)/p + 1)*16;	// часть N для 1 нити
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
		if (tab & 1)
			transaT( A + i, K, M, abu + i*K, 6, m );
		else
			transband( A + i*K, K, m, abu + i*K, 6 );
	}
	if (tab & 8) {
		JOB_GEMM jg = {tab, M, N, K, abu, B, C};
		t_run( rob_gemm, &jg );
	} else
		gemm2j( tab, M, N, K, abu, M, B, N, C, N );
	_mm_free( abu );
}
