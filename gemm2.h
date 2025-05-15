#pragma once

enum {		// tab bits
	TAB_TRANSP_A = 1,	// A transponed 
	TAB_TRANSP_B = 2,	// B transponed
	TAB_ZEROED_C = 4,	// C=0 before (no need to zero in gemm)
	TAB_PARALLEL = 8,	// multithreaded gemm
};

inline int _min( int a, int b ) { return a < b? a : b; }

void gemm2( int tab, int M, int N, int K, const float *A, const float *B, float *C );

struct JOB_GEMM { int tab, M, N, K; const float *A, *B; float *C; };

