/*
	держатель рабочих лошадей (simple case + plain C)
*/
#pragma once

enum {MAXPOT=16, ASK=0, WAIT, FORCE,};

struct troika;
struct job {
	struct troika *t;	// -> self
	int ij;				// порядковый номер задания (0..nya-1)
	int ret;			// ret = rob( job* );
};

struct troika {
	int nya;				// число ядер
	DWORD gran;				// page granularity
	struct job mjob[MAXPOT];
	HANDLE mthr[MAXPOT];	// заранее созданные потоки (или ждут отмашки, или трудятся)
	HANDLE green[MAXPOT];	// есть работа потоку (авто, 0)
	HANDLE vacant;			// 1 если можно его занять
	HANDLE ready;			// 1 если можно вся работа сделана
	int nthr;				// MAXPOT >= число потоков >= nya
	LONG njob;				// сколько из них еще работают
	short tostop;			// общий сигнал остановки
	// надо задавать клиенту, можно через help_cli
	void *host;				// клиент
	int (*rob)(void *host, int j);	// дается пользователем для выполнения его задач
	void (*after)( void *host );	// звать после окончания всех потоков
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
