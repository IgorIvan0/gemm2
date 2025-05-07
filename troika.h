/*
	��������� ������� ������� (simple case + plain C)
*/
#pragma once

enum {MAXPOT=16, ASK=0, WAIT, FORCE,};

struct troika;
struct job {
	struct troika *t;	// -> self
	int ij;				// ���������� ����� ������� (0..nya-1)
	int ret;			// ret = rob( job* );
};

struct troika {
	int nya;				// ����� ����
	DWORD gran;				// page granularity
	struct job mjob[MAXPOT];
	HANDLE mthr[MAXPOT];	// ������� ��������� ������ (��� ���� �������, ��� ��������)
	HANDLE green[MAXPOT];	// ���� ������ ������ (����, 0)
	HANDLE vacant;			// 1 ���� ����� ��� ������
	HANDLE ready;			// 1 ���� ����� ��� ������ �������
	int nthr;				// MAXPOT >= ����� ������� >= nya
	LONG njob;				// ������� �� ��� ��� ��������
	short tostop;			// ����� ������ ���������
	// ���� �������� �������, ����� ����� help_cli
	void *host;				// ������
	int (*rob)(void *host, int j);	// ������ ������������� ��� ���������� ��� �����
	void (*after)( void *host );	// ����� ����� ��������� ���� �������
#ifdef __cplusplus
	int get_result( int i ) { return mjob[i].ret; }
#endif
};

void work_over( struct troika *tr, int timeout );
int t_init( struct troika *tr, int n );
int push_all( struct troika *tr, int (*rob)(void *host, int j), void (*after)(void *host) = NULL, int waits = 0 );
int push_all2( struct troika *tr, int (*rob)(void *host, int j), void *host, int waits = 0 );

int t_run( int (*rob)(void *host, int j), void *host, int waits = 5000 );
int t_wait( int waits );
int t_nya();
