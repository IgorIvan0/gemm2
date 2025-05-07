#include "ugemm.h"
#include "gemm2.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

float maxdif( const float *A, const float *B, int n ) {
	float d, d0 = 0;
	for ( ; --n >= 0; A++, B++) 
		if ((d = fabsf(*A - *B)) > d0) 
			d0 = d;
	return d0;
}

#ifdef _DEBUG
const int NITER = 1;
#else
const int NITER = 10;
#endif
int main() {
	int M = 1000, N = 1200, K = 1150;
	int i, na = M*K, nb = K*N, nc = M*N;
	float alpha = 1, beta  = 0;

	float *a = (float*)malloc_a( (na + nb)*sizeof(a[0]), 32 ), *b = a + na;
	for (i = na + nb; --i >= 0; ) { a[i] = (float)rand()/RAND_MAX; }
	float *c = (float*)malloc_a( 9*nc*sizeof(c[0]), 32 ), *s = c + 3*nc, *x = s + 3*nc;

	sgemm_cpu( 'R', 'N', 'N', M, N, K, alpha, a,K, b,N, beta, c,N );
	sgemm_cpu( 'R', 'N', 'T', M, N, K, alpha, a,K, b,K, beta, c + nc,N );
	sgemm_cpu( 'R', 'T', 'N', M, N, K, alpha, a,M, b,N, beta, c + 2*nc,N );

	uint64_t t0 = __rdtsc();

	for (i = NITER; --i >= 0; ) { sgemm_sse( 'R', 'N', 'N', M, N, K, alpha, a,K, b,N, beta, s,N ); }
	uint64_t ts1 = __rdtsc() - t0; t0 += ts1;

	for (i = NITER; --i >= 0; ) { sgemm_sse( 'R', 'N', 'T', M, N, K, alpha, a,K, b,K, beta, s + nc,N ); }
	uint64_t ts2 = __rdtsc() - t0; t0 += ts2;

	for (i = NITER; --i >= 0; ) { sgemm_sse(  'R', 'T', 'N', M, N, K, alpha, a,M, b,N, beta, s + 2*nc,N  ); }
	uint64_t ts3 = __rdtsc() - t0; t0 += ts3;

	for (i = NITER; --i >= 0; ) { gemm2( 0, M, N, K, a, b, x ); }
	uint64_t t1 = __rdtsc() - t0; t0 += t1;

	for (i = NITER; --i >= 0; ) { gemm2( 2, M, N, K, a, b, x + nc ); }
	uint64_t t2 = __rdtsc() - t0; t0 += t2;

	for (i = NITER; --i >= 0; ) { gemm2( 1, M, N, K, a, b, x + 2*nc ); }
	uint64_t t3 = __rdtsc() - t0; t0 += t2;

	float e1 = maxdif( x, c, nc ), e2 = maxdif( x + nc, c + nc, nc ), e3 = maxdif( x + 2*nc, c + 2*nc, nc );
	double kt = 2.0*NITER*M*N*K;	// nflops
	printf( "%3.3lf(%3.5f) %3.3lf(%3.5f) %3.3lf(%3.5f)\n", kt/t1, e1, kt/t2, e2, kt/t3, e3 );
	FILE *f = 0; fopen_s( &f, "gemm2.log", "ab" );
	fprintf( f, "%3.3lf(%3.5f) %3.3lf(%3.5f) %3.3lf(%3.5f)\n", kt/t1, e1, kt/t2, e2, kt/t3, e3 );
	fclose( f );

	e1 = maxdif( s, c, nc ), e2 = maxdif( s + nc, c + nc, nc ), e3 = maxdif( s + 2*nc, c + 2*nc, nc );
	printf( "%3.3lf(%3.5f) %3.3lf(%3.5f) %3.3lf(%3.5f)\n", kt/ts1, e1, kt/ts2, e2, kt/ts3, e3 );

	free_a(a);
	free_a(c);

	return 0;
}
