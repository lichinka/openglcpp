
/* Author: Gerhard Hoffmann                                                  */
/* Basic implementation of Keccak-f1600. Calculates with a file as input     */
/* the corresonding digest (of theoretical infinite length).                 */
/*                                                                           */
/* The corresonding kernel(s) are executed by only one warp (= 32 threads),  */
/* and here only 25 of them do real work. Because threads inside a warp are  */
/* always synchronized there is not synchronization problem.                 */
/*                                                                           */
/* However, we have a high number of bank conflicts. This is due to the      */
/* following facts: on one hand Keccak-f1600 is 64-bit by default, but the   */
/* 16 memory banks of the GTX 295 consist of 1024 integers (of 32 bit) each. */
/* On the other hand, due to the access pattern of Keccak there is  always a */
/* high probability of two (or more threads) accessing the same memory bank. */
/* There is a modified 32-bit version of Keccak-f1600. Using adapted tables  */
/* it would look something like                                              */
/*                                                                           */
/*  for(int i=0;i<ROUNDS;++i) {                                              */ 
/*      C[t].x = A[s].x^A[s+5].x^A[s+10].x^A[s+15].x^A[s+20].x;              */
/*      C[t].y = A[s].y^A[s+5].y^A[s+10].y^A[s+15].y^A[s+20].y;              */
/*      D[t].x = C[b[20+s]].x^R32(C[b[5+s]].y,1,31);                         */
/*      D[t].y = C[b[20+s]].y^C[b[5+s]].x;                                   */
/*      C[t].x = R32(A[a[t]].x^D[b[t]].x, ro[0][t][0], ro[0][t][1]);         */
/*      C[t].y = R32(A[a[t]].y^D[b[t]].y, ro[1][t][0], ro[1][t][1]);         */
/*      A[d[t]].x = C[c[t][0]].x ^ ((~C[c[t][1]].x) & C[c[t][2]].x);         */
/*      A[d[t]].y = C[c[t][0]].y ^ ((~C[c[t][1]].y) & C[c[t][2]].y);         */
/*      A[t].x ^= rc[0][(t==0) ? 0 : 1][i];                                  */
/*      A[t].y ^= rc[1][(t==0) ? 0 : 1][i];                                  */
/*  }                                                                        */
/*                                                                           */
/* and profiling the resulting code has shown no advantage compared to the   */
/* pure 64-bit version. It might be possible to use this 32-bit version and  */
/* combine it with texture memory as caching device. This issue is left open */
/* for now.                                                                  */
/*                                                                           */
/* Finally, we use only one warp for this basic case, which means an         */
/* occupancy of only 0.031. (This is not a typical use case of the GPU.)     */
/*                                                                           */
/* The timing results of this basic version are therefore not really         */
/* amazing. To hash a 1MB file takes roughly 0.5 secs for a digest length    */
/* smaller than 1024.                                                        */
/*                                                                           */
/* Some values reported by the CUDA profiler:                                */
/*                                                                           */
/* File size [bytes] | Digest length [bits] |  Time [nanoseconds]            */
/* ------------------+----------------------+----------------------          */
/*              3510 |                 100  |            1650752             */
/*            810035 |                 100  |          370540406             */
/*           1287583 |                 100  |          588969375             */
/*           8223936 |                 100  |         3761311000             */
/*          53872640 |                 100  |        24638936000             */
/**/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <errno.h>
#include <cutil_inline.h>
#include <cuda_keccak_basic.cuh>

static uint8_t  *h_data;
static uint64_t *d_data;
static uint8_t  *h_out;
static uint64_t *d_out;

#define PRINT_GPU_RESULT           \
    printf("\nOutput of GPU:\n");  \
    for(int i=0;i<200;++i) {       \
        printf("%02X ", h_out[i]); \
    } printf("\n\n\n"); 

#define MAX_FILE_SIZE 500000000
#define BITRATE       1024
#define ROUNDS        24 
#define OFFSET        63 
#define R64(a,b,c) (((a) << b) ^ ((a) >> c)) /* works on the GPU also for 
                                                b = 64 or c = 64 */
static const uint64_t round_const[5][ROUNDS] = {
    {0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
     0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
     0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
     0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
     0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
     0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
     0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
     0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL}};

/* Rho-Offsets. Note that for each entry pair their respective sum is 64.
   Only the first entry of each pair is a rho-offset. The second part is
   used in the R64 macros. */
static const int rho_offsets[25][2] = {
       /*y=0*/         /*y=1*/         /*y=2*/         /*y=3*/         /*y=4*/
/*x=0*/{ 0,64}, /*x=1*/{44,20}, /*x=2*/{43,21}, /*x=3*/{21,43}, /*x=4*/{14,50},
/*x=1*/{ 1,63}, /*x=2*/{ 6,58}, /*x=3*/{25,39}, /*x=4*/{ 8,56}, /*x=0*/{18,46},
/*x=2*/{62, 2}, /*x=3*/{55, 9}, /*x=4*/{39,25}, /*x=0*/{41,23}, /*x=1*/{ 2,62},
/*x=3*/{28,36}, /*x=4*/{20,44}, /*x=0*/{ 3,61}, /*x=1*/{45,19}, /*x=2*/{61, 3},
/*x=4*/{27,37}, /*x=0*/{36,28}, /*x=1*/{10,54}, /*x=2*/{15,49}, /*x=3*/{56, 8}};

static const int a_host[25] = {
    0,  6, 12, 18, 24,
    1,  7, 13, 19, 20,
    2,  8, 14, 15, 21,
    3,  9, 10, 16, 22,
    4,  5, 11, 17, 23};

static const int b_host[25] = {
    0,  1,  2,  3, 4,
    1,  2,  3,  4, 0,
    2,  3,  4,  0, 1,
    3,  4,  0,  1, 2,
    4,  0,  1,  2, 3};
    
static const int c_host[25][3] = {
    { 0, 1, 2}, { 1, 2, 3}, { 2, 3, 4}, { 3, 4, 0}, { 4, 0, 1},
    { 5, 6, 7}, { 6, 7, 8}, { 7, 8, 9}, { 8, 9, 5}, { 9, 5, 6},
    {10,11,12}, {11,12,13}, {12,13,14}, {13,14,10}, {14,10,11},
    {15,16,17}, {16,17,18}, {17,18,19}, {18,19,15}, {19,15,16},
    {20,21,22}, {21,22,23}, {22,23,24}, {23,24,20}, {24,20,21}};
    
static const int d_host[25] = {
          0,  1,  2,  3,  4,
         10, 11, 12, 13, 14,
         20, 21, 22, 23, 24,
          5,  6,  7,  8,  9,
         15, 16, 17, 18, 19};

__device__ __constant__ uint32_t a[25];
__device__ __constant__ uint32_t b[25];
__device__ __constant__ uint32_t c[25][3];
__device__ __constant__ uint32_t d[25];
__device__ __constant__ uint32_t ro[25][2];
__device__ __constant__ uint64_t rc[5][ROUNDS];

__global__
void keccac_squeeze_kernel(uint64_t *data) {/* In case a digest of length   */
    int const t = threadIdx.x;   /* greater than 1024 bit is needed, call   */           
    int const s = threadIdx.x%5; /*                   kernel multiple times.*/

    __shared__ uint64_t A[25];  
    __shared__ uint64_t C[25]; 
    __shared__ uint64_t D[25]; 

    if(t < 25) {
        A[t] = data[t];

        for(int i=0;i<ROUNDS;++i) {                             /* Keccak-f */
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }

        data[t] = A[t];
    }
}

__global__
void keccac_kernel(uint64_t *data, uint64_t *out, uint64_t databitlen) {

    int const t = threadIdx.x; 
    int const s = threadIdx.x%5;

    __shared__ uint64_t A[25];  
    __shared__ uint64_t B[25];  
    __shared__ uint64_t C[25]; 
    __shared__ uint64_t D[25]; 

    if(t < 25) {
        A[t] = 0ULL;
        B[t] = 0ULL;
        if(t < 16) 
            B[t] = data[t]; 

        int const blocks = databitlen/BITRATE;
       
        for(int block=0;block<blocks;++block) { 

            A[t] ^= B[t];

            data += BITRATE/64;
            if(t < 16) B[t] = data[t];       /* prefetch data */

            for(int i=0;i<ROUNDS;++i) { 
                C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
                D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
                C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
                A[t] ^= rc[(t==0) ? 0 : 1][i]; 
            }

            databitlen -= BITRATE;
        }

        int const bytes = databitlen/8;/*bytes will be smaller than BITRATE/8*/

        if(t == 0) {
            uint8_t *p = (uint8_t *)B+bytes;
            uint8_t const q = *p;
            *p++ = (q >> (8-(databitlen&7)) | (1 << (databitlen&7)));
            *p++ = 0x00; 
            *p++ = BITRATE/8; 
            *p++ = 0x01; 
            while(p < (uint8_t *)&B[25])
                *p++ = 0;
        }

        if(t < 16) A[t] ^= B[t];

        for(int i=0;i<ROUNDS;++i) { 
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }

        if((bytes+4) > BITRATE/8) {/* then thread 0 has crossed the 128 byte */
            if(t < 16) B[t] = 0ULL;/* boundary and touched some higher parts */
            if(t <  9) B[t] = B[t+16]; /* of B.                              */
            if(t < 16) A[t] ^= B[t];
            
            for(int i=0;i<ROUNDS;++i) { 
                C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
                D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
                C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
                A[t] ^= rc[(t==0) ? 0 : 1][i]; 
            }
        } 

        out[t] = A[t];
    }
}

void call_keccak_basic_kernel(char const *filename, int digestlength) {

    struct stat buf;
    size_t size;
 
    if(stat(filename, &buf) < 0) {
        fprintf(stderr, "stat %s failed: %s\n", filename, strerror(errno));
        return;
    } 

    if(buf.st_size == 0 || buf.st_size > MAX_FILE_SIZE) {
        fprintf(stderr, "%s wrong sized %d\n", filename, (int)buf.st_size);
        return;
    }
                                        /* align the data on BITRATE/8 bytes */
    size = ((buf.st_size-1)/(BITRATE/8) + 1)*(BITRATE/8);

    h_data = (uint8_t *)calloc(1, size);
    h_out = (uint8_t *)malloc(200);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, size));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_out, 200));

    FILE *in = fopen(filename, "r");

    if(in == NULL) {
        fprintf(stderr, "open %s failed: %s\n", filename, strerror(errno));
        return;
    }

    memset(h_data, 0x00, size);  
    /* read in the document to be hashed */
    if(fread(h_data, 1, (size_t)buf.st_size, in) < buf.st_size) {
        fprintf(stderr, "read %s failed: %s\n", filename, strerror(errno));
        return;
    }

    fclose(in);

    int count = 0;
    for(int i=0;i<8;++i)
        if((h_data[buf.st_size-1] >> i) & 1) { 
            count = 8 - i; break;
        }

    /* copy the tables from host to GPU */
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(a, a_host, sizeof(a_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(b, b_host, sizeof(b_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c, c_host, sizeof(c_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d, d_host, sizeof(d_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ro, rho_offsets, sizeof(rho_offsets)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rc, round_const, sizeof(round_const)));

    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    keccac_kernel<<<1,32>>>(d_data, d_out, (buf.st_size-1)*8 + count);

    memset(h_out, 0x00, 200);
    /* fetch the generated data*/
    CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));

    PRINT_GPU_RESULT;

    for(int i=0;i<digestlength/BITRATE;++i) {
        keccac_squeeze_kernel<<<1,32>>>(d_out);
        CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));
        PRINT_GPU_RESULT;
    }

    CUDA_SAFE_CALL(cudaFree(d_data));
    CUDA_SAFE_CALL(cudaFree(d_out));
    free(h_data);
    free(h_out);
}


