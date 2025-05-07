#pragma once

inline int _min( int a, int b ) { return a < b? a : b; }

void gemm2( int tab, int M, int N, int K, const float *A, const float *B, float *C );

struct JOB_GEMM { int tab, M, N, K; const float *A, *B; float *C; };

