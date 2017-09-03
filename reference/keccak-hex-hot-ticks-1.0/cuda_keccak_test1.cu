
/* Author: Gerhard Hoffmann */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <cutil_inline.h>
#include <cuda_keccak.cuh>

#define ROUNDS 24 
#define OFFSET 63 
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
void keccac_kernel(uint64_t *d_data) {

    int const t = threadIdx.x; 
    int const s = threadIdx.x%5;

    __shared__ uint64_t A[25];  
    __shared__ uint64_t C[25]; 
    __shared__ uint64_t D[25]; 

    if(t < 25) { 
        A[t] = d_data[t];
   
        /*******************************************************************/
        /*                         KECCAK                                  */
        /*******************************************************************/
        /* the Keccak-f function: it consists of 5 functions named chi, pi,*/
        /* theta, rho and iota. The following 5 lines are a compact version*/
        /* of those functions.                                             */
        for(int i=0;i<ROUNDS;++i) { 

            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);

            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);

            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }

        d_data[t] = A[t];

        for(int i=0;i<ROUNDS;++i) { 

            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);

            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);

            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }

        d_data[t+25] = A[t];
    }
}

uint64_t *h_data;
uint64_t *d_data;

#define DATA_SIZE 50

void call_keccak_kernel() {

    /* copy the tables from host to GPU */
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(a, a_host, sizeof(a_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(b, b_host, sizeof(b_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c, c_host, sizeof(c_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d, d_host, sizeof(d_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ro, rho_offsets, sizeof(rho_offsets)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rc, round_const, sizeof(round_const)));

    CUDA_SAFE_CALL(cudaMallocHost((void **)&h_data,DATA_SIZE*sizeof(*h_data)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_data,DATA_SIZE*sizeof(*d_data)));

    for(int i=0;i<50;++i) {
        h_data[i] = 0ULL;
    } 

    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, DATA_SIZE*sizeof(*d_data)
                 , cudaMemcpyHostToDevice));

    keccac_kernel<<<1,32>>>(d_data);

    /* fetch the generated data*/
    CUDA_SAFE_CALL(cudaMemcpy(h_data, d_data, DATA_SIZE*sizeof(*d_data)
                 , cudaMemcpyDeviceToHost));

    printf("Output of reference implementation ...\n");
    printf("After iota:\n");
    printf("F1258F7940E1DDE7 84D5CCF933C0478A D598261EA65AA9EE BD1547306F80494D 8B284E056253D057\n");
    printf("FF97A42D7F8E6FD4 90FEE5A0A44647C4 8C5BDA0CD6192E76 AD30A6F71B19059C 30935AB7D08FFC64\n");
    printf("EB5AA93F2317D635 A9A6E6260D712103 81A57C16DBCF555F 43B831CD0347C826 01F22F1A11A5569F\n");
    printf("05E5635A21D9AE61 64BEFEF28CC970F2 613670957BC46611 B87C5A554FD00ECB 8C3EE88A1CCF32C8\n");
    printf("940C7922AE3A2614 1841F924A2C509E4 16F53526E70465C2 75F644E97F30A13B EAF1FF7B5CECA249\n");
    
    printf("\nOutput of GPU:\n");
    for(int i=0;i<25;++i) {
        if(i%5 == 0) printf("\n");
        printf("%016lX ", h_data[i]);
    }printf("\n\n\n");

    printf("Input of permutation:\n");
    printf("F1258F7940E1DDE7 84D5CCF933C0478A D598261EA65AA9EE BD1547306F80494D 8B284E056253D057\n");
    printf("FF97A42D7F8E6FD4 90FEE5A0A44647C4 8C5BDA0CD6192E76 AD30A6F71B19059C 30935AB7D08FFC64\n");
    printf("EB5AA93F2317D635 A9A6E6260D712103 81A57C16DBCF555F 43B831CD0347C826 01F22F1A11A5569F\n");
    printf("05E5635A21D9AE61 64BEFEF28CC970F2 613670957BC46611 B87C5A554FD00ECB 8C3EE88A1CCF32C8\n");
    printf("940C7922AE3A2614 1841F924A2C509E4 16F53526E70465C2 75F644E97F30A13B EAF1FF7B5CECA249\n");

    printf("\nOutput of reference implementation ...\n");
    printf("After iota:\n");
    printf("2D5C954DF96ECB3C 6A332CD07057B56D 093D8D1270D76B6C 8A20D9B25569D094 4F9C4F99E5E7F156\n");
    printf("F957B9A2DA65FB38 85773DAE1275AF0D FAF4F247C3D810F7 1F1B9EE6F79A8759 E4FECC0FEE98B425\n");
    printf("68CE61B6B9CE68A1 DEEA66C4BA8F974F 33C43D836EAFB1F5 E00654042719DBD9 7CF8A9F009831265\n");
    printf("FD5449A6BF174743 97DDAD33D8994B40 48EAD5FC5D0BE774 E3B8C8EE55B7B03C 91A0226E649E42E9\n");
    printf("900E3129E7BADD7B 202A9EC5FAA3CCE8 5B3402464E1C3DB6 609F4E62A44C1059 20D06CD26A8FBF5C\n");

    printf("\nOutput of GPU:\n");
    for(int i=25;i<DATA_SIZE;++i) {
        if(i%5 == 0) printf("\n");
        printf("%016lX ", h_data[i]);
    }printf("\n");

    CUDA_SAFE_CALL(cudaFree(d_data));
    CUDA_SAFE_CALL(cudaFreeHost(h_data));
}


