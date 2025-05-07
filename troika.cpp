/*
	��������� ������� �������. ������ ������ ������������ �-���
	int (*rob)( void *host, int j );	// (n) ��������� ����� j �� n ��� ������ ������
	int (*after)( void *host );	// (1) ���������� ���������� ������ (optional, may be NULL)
	� ������ ����� void *host ��� �������� � join, ������ ��� ��� ������ ����� ����� 
	push_all( troika, rob, after, waits ) ��� push_all2( troika, rob, host, waits )
*/
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "troika.h"

// ��������� ������ ������� (�� ����� �� ���������������)
void work_over( struct troika *tr, int timeout ) {
	tr->tostop = 3;	// ������ ��������� ���������� ������
	for ( ; tr->nthr > 0; ) {
		SetEvent( tr->green[--tr->nthr] );
		WaitForSingleObject( tr->mthr[tr->nthr], timeout );
		CloseHandle( tr->mthr[tr->nthr] );
		CloseHandle( tr->green[tr->nthr] );
	}
	CloseHandle( tr->vacant );
	CloseHandle( tr->ready );
	memset( tr, 0, sizeof(*tr) );
}

// ��� �������� 1 �����
static DWORD rob1t( struct troika *tr, struct job *z ) {
	for ( ; !(tr->tostop & 2); ) {
		if (WaitForSingleObject( tr->green[z->ij], INFINITE ) != WAIT_OBJECT_0 || tr->tostop) break;	// ���� ������� � ������
		z->ret = (*tr->rob)( tr->host, z->ij );		// ������ ���� ����� ������
		if (InterlockedDecrement( &tr->njob ) > 0) continue;	// ���� ��� �� ��������
		if (tr->after)		// ��������� ����� �-��� �����������
			(*(tr->after))( tr->host );
		SetEvent( tr->ready );
		SetEvent( tr->vacant );
	}
	return z->ret;
}
static DWORD WINAPI rob_proc( void *foo ) {
	struct job *z = (struct job*)foo;
	return rob1t( z->t, z );
}

// ����� ������ ��� (�� ����� �� ����������������)
int t_init( struct troika *tr, int n ) {
	SYSTEM_INFO si;
	DWORD i, threadid;
	if (tr->nya) return 1;
	memset( tr, 0, sizeof(*tr) ) ;
	GetSystemInfo( &si );
	tr->nya = min( MAXPOT, si.dwNumberOfProcessors );
	tr->gran = si.dwAllocationGranularity;
	if (n < tr->nya) n = tr->nya;
	for (i = 0; i < (DWORD)n; i++) {
		tr->mjob[i].t = tr;
		tr->mjob[i].ij = i;
		tr->mjob[i].ret = 0;
		if (!(tr->green[i] = CreateEvent( NULL, FALSE/*=AUTO*/, 0, NULL ))) return -1;
		if (!(tr->mthr[i] = CreateThread( NULL, 0, &rob_proc, (void*)&tr->mjob[i], 0, &threadid ))) return -1;
	}
	tr->nthr = i;
	if (!(tr->vacant = CreateEvent( NULL, FALSE/*=AUTO*/, 1, NULL ))) return -1;
	if (!(tr->ready = CreateEvent( NULL, TRUE/*=MANU*/, 1, NULL ))) return -1;
	return 0;
}

int push_all( struct troika *tr, int (*rob)(void *host, int j), void (*after)(void *host), int waits ) {
	WaitForSingleObject( tr->vacant, INFINITE );
	ResetEvent( tr->ready );
	tr->rob = rob;
	tr->after = after;
	tr->njob = tr->nthr;
	for (int i = 0; i < tr->nthr; i++)
		SetEvent( tr->green[i] );
	return waits? WaitForSingleObject( tr->ready, waits ) : 0;
}

int push_all2( struct troika *tr, int (*rob)(void *host, int j), void *host, int waits ) {
	if (!tr->nya)
		t_init( tr, 0 );
	tr->host = host;
	return push_all( tr, rob, NULL, waits );
}

static troika T;

int t_nya() { 
	if (!T.nya)
		t_init( &T, 0 );
	return T.nya; 
}

int t_run( int (*rob)(void *host, int j), void *host, int waits ) {
	return push_all2( &T, rob, host, waits );
}

int t_wait( int waits ) {
	return WaitForSingleObject( T.ready, waits );
}
