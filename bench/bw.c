#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
static uint8_t *buf; static size_t N = 1UL << 30; static int T;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void *rd(void *a){ long t=(long)a; size_t lo=N*t/T, hi=N*(t+1)/T; __m256i acc=_mm256_setzero_si256();
  for(size_t i=lo;i<hi;i+=128){ acc=_mm256_add_epi64(acc,_mm256_load_si256((void*)(buf+i))); acc=_mm256_add_epi64(acc,_mm256_load_si256((void*)(buf+i+32)));
    acc=_mm256_add_epi64(acc,_mm256_load_si256((void*)(buf+i+64))); acc=_mm256_add_epi64(acc,_mm256_load_si256((void*)(buf+i+96)));}
  return (void*)(long)_mm256_extract_epi64(acc,0);}
int main(){ buf=aligned_alloc(64,N); memset(buf,1,N);
  for(T=1;T<=12;T+= (T<4?1:(T<8?2:4))){ pthread_t th[16]; double best=1e9;
    for(int r=0;r<4;r++){ double s=now(); for(long t=0;t<T;t++) pthread_create(&th[t],0,rd,(void*)t); for(int t=0;t<T;t++) pthread_join(th[t],0); double d=now()-s; if(d<best)best=d;}
    printf("threads %2d: %.1f GB/s\n",T,N/best/1e9);} }
