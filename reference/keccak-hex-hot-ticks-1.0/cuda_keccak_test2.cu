
/* Author: Gerhard Hoffmann */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <cutil_inline.h>
#include <cuda_keccak.cuh>

static uint8_t  *h_data;
static uint64_t *d_data;
static uint8_t  *h_out;
static uint64_t *d_out;

#define PRINT_GPU_RESULT           \
    printf("\nOutput of GPU:\n");  \
    for(int i=0;i<200;++i) {       \
        printf("%02X ", h_out[i]); \
    } printf("\n\n\n"); 

#define PRINT_CPU_INPUT(length,message)                                \
    printf("\nOutput of reference implementation for input message =");\
    for(int i=0;i<length;++i) {                                        \
        if(i%16 == 0) printf("\n");                                    \
        printf("\\x%02X", (uint8_t)message[i]);                        \
    } printf("\n\n");

#define DATA_SIZE  384 
#define BITRATE    1024
#define ROUNDS     24 
#define OFFSET     63 
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
void keccac_squeeze_kernel(uint64_t *data) {
    int const t = threadIdx.x; 
    int const s = threadIdx.x%5;

    __shared__ uint64_t A[25];  
    __shared__ uint64_t C[25]; 
    __shared__ uint64_t D[25]; 

    if(t < 25) {
        A[t] = data[t];

        for(int i=0;i<ROUNDS;++i) { 
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
        int block;
       
        for(block=0;block<blocks;++block) {  /* Step 0: load without padding */

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


void call_keccak_kernel() {

    /* copy the tables from host to GPU */
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(a, a_host, sizeof(a_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(b, b_host, sizeof(b_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c, c_host, sizeof(c_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d, d_host, sizeof(d_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ro, rho_offsets, sizeof(rho_offsets)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rc, round_const, sizeof(round_const)));

    CUDA_SAFE_CALL(cudaMallocHost((void **)&h_data,DATA_SIZE*sizeof(*h_data)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_data,DATA_SIZE/8*sizeof(*d_data)));
    CUDA_SAFE_CALL(cudaMallocHost((void **)&h_out,DATA_SIZE*sizeof(*h_out)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_out,DATA_SIZE/8*sizeof(*d_out)));

    char const *message1 = "\x53\x58\x7B\xC8";
    unsigned int message1Length = 29;
    char const *message2 =
        "\x83\xAF\x34\x27\x9C\xCB\x54\x30\xFE\xBE\xC0\x7A\x81\x95\x0D\x30"
        "\xF4\xB6\x6F\x48\x48\x26\xAF\xEE\x74\x56\xF0\x07\x1A\x51\xE1\xBB"
        "\xC5\x55\x70\xB5\xCC\x7E\xC6\xF9\x30\x9C\x17\xBF\x5B\xEF\xDD\x7C"
        "\x6B\xA6\xE9\x68\xCF\x21\x8A\x2B\x34\xBD\x5C\xF9\x27\xAB\x84\x6E"
        "\x38\xA4\x0B\xBD\x81\x75\x9E\x9E\x33\x38\x10\x16\xA7\x55\xF6\x99"
        "\xDF\x35\xD6\x60\x00\x7B\x5E\xAD\xF2\x92\xFE\xEF\xB7\x35\x20\x7E"
        "\xBF\x70\xB5\xBD\x17\x83\x4F\x7B\xFA\x0E\x16\xCB\x21\x9A\xD4\xAF"
        "\x52\x4A\xB1\xEA\x37\x33\x4A\xA6\x64\x35\xE5\xD3\x97\xFC\x0A\x06"
        "\x5C\x41\x1E\xBB\xCE\x32\xC2\x40\xB9\x04\x76\xD3\x07\xCE\x80\x2E"
        "\xC8\x2C\x1C\x49\xBC\x1B\xEC\x48\xC0\x67\x5E\xC2\xA6\xC6\xF3\xED"
        "\x3E\x5B\x74\x1D\x13\x43\x70\x95\x70\x7C\x56\x5E\x10\xD8\xA2\x0B"
        "\x8C\x20\x46\x8F\xF9\x51\x4F\xCF\x31\xB4\x24\x9C\xD8\x2D\xCE\xE5"
        "\x8C\x0A\x2A\xF5\x38\xB2\x91\xA8\x7E\x33\x90\xD7\x37\x19\x1A\x07"
        "\x48\x4A\x5D\x3F\x3F\xB8\xC8\xF1\x5C\xE0\x56\xE5\xE5\xF8\xFE\xBE"
        "\x5E\x1F\xB5\x9D\x67\x40\x98\x0A\xA0\x6C\xA8\xA0\xC2\x0F\x57\x12"
        "\xB4\xCD\xE5\xD0\x32\xE9\x2A\xB8\x9F\x0A\xE1";
    unsigned int message2Length = 2008;
    char const *message3 =
        "\x83\xAF\x34\x27\x9C\xCB\x54\x30\xFE\xBE\xC0\x7A\x81\x95\x0D\x30"
        "\xF4\xB6\x6F\x48\x48\x26\xAF\xEE\x74\x56\xF0\x07\x1A\x51\xE1\xBB"
        "\xC5\x55\x70\xB5\xCC\x7E\xC6\xF9\x30\x9C\x17\xBF\x5B\xEF\xDD\x7C"
        "\x6B\xA6\xE9\x68\xCF\x21\x8A\x2B\x34\xBD\x5C\xF9\x27\xAB\x84\x6E"
        "\x38\xA4\x0B\xBD\x81\x75\x9E\x9E\x33\x38\x10\x16\xA7\x55\xF6\x99"
        "\xDF\x35\xD6\x60\x00\x7B\x5E\xAD\xF2\x92\xFE\xEF\xB7\x35\x20\x7E"
        "\xBF\x70\xB5\xBD\x17\x83\x4F\x7B\xFA\x0E\x16\xCB\x21\x9A\xD4\xAF"
        "\x52\x4A\xB1\xEA\x37\x33\x4A\xA6\x64\x35\xE5\xD3\x97\xFC\x0A\x06"
        "\x5C\x41\x1E\xBB\xCE\x32\xC2\x40\xB9\x04\x76\xD3\x07\xCE\x80\x2E"
        "\xC8\x2C\x1C\x49\xBC\x1B\xEC\x48\xC0\x67\x5E\xC2\xA6\xC6\xF3\xED"
        "\x3E\x5B\x74\x1D\x13\x43\x70\x95\x70\x7C\x56\x5E\x10\xD8\xA2\x0B"
        "\x8C\x20\x46\x8F\xF9\x51\x4F\xCF\x31\xB4\x24\x9C\xD8\x2D\xCE\xE5"
        "\x8C\x0A\x2A\xF5\x38\xB2\x91\xA8\x7E\x33\x90\xD7\x37\x19\x1A\x07"
        "\x48\x4A\x5D\x3F\x3F\xB8\xC8\xF1\x5C\xE0\x56\xE5\xE5\xF8\xFE\xBE"
        "\x5E\x1F\xB5\x9D\x67\x40\x98\x0A\xA0\x6C\xA8\xA0\xC2\x0F\x57\x12"
        "\xB4\xCD\xE5\xD0\x32\xE9\x2A\xB8\x9F\x0A\xE1\xFF\xFF\xFF\xFF\xFF";
    unsigned int message3Length = 2048;

    uint64_t *h = (uint64_t *)&h_data[0];
    for(int i=0;i<DATA_SIZE/8;++i) {
        h[i] = 0ULL;
    }

    memset(h_data, 0x00, DATA_SIZE*sizeof(*h_data));
    memcpy(h_data, message1, (message1Length-1)/8 + 1); 

    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, DATA_SIZE/8*sizeof(*d_data)
                 , cudaMemcpyHostToDevice));

    keccac_kernel<<<1,32>>>(d_data, d_out, message1Length);

    /* fetch the generated data*/
    CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));

    PRINT_CPU_INPUT((message1Length-1)/8+1, message1);

    printf("State after permutation:\n"); /* print CPU reference result */
    printf("DE EB FB 5F BC 67 14 3A 70 F5 EE 51 8F 3C E2 0A 70 2A 3C 25 "
           "0C 22 D9 39 D7 EE F5 98 A3 9C A5 C5 37 41 B6 F5 7B 58 40 AD "
           "D2 8E F6 14 0A AD 9D 4C 2B 8E CC 6A 89 FC 5E FE 73 1F 5E 69 "
           "7B 83 B8 1C 27 ED E0 D2 26 BB 30 DE 0A 93 F5 CE DB C1 6E 32 "
           "BA 9D 6B 10 48 8A 5A 0E 55 5C B2 96 9F 51 E5 8D 46 F0 03 F5 "
           "0F 9D 84 5A AF 43 07 66 76 23 82 AD FD 9B 4C F0 59 16 DF D6 "
           "5C 8A 8C FC DE C5 D0 45 35 D0 D5 37 B1 E0 F8 37 5F FF 45 9D "
           "E4 F9 80 63 A8 C3 29 94 FF 19 85 6B B8 22 64 78 05 2C 8D 6F "
           "55 B3 8A 2B 01 12 4A D7 C8 AA D1 5D A6 C8 C7 18 90 94 8E E0 "
           "98 7B E1 F2 04 D2 91 C4 D6 DA D5 A2 33 27 42 2C 7B CD 3E D4\n"); 

    PRINT_GPU_RESULT;

    memset(h_data, 0x00, DATA_SIZE*sizeof(*h_data));
    //memcpy(h_data, message2, (message2Length-1)/8 + 1); 
    memcpy(h_data, message2, 251); 
    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, DATA_SIZE/8*sizeof(*d_data)
                 , cudaMemcpyHostToDevice));

    keccac_kernel<<<1,32>>>(d_data, d_out, message2Length);

    CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));

    PRINT_CPU_INPUT((message2Length-1)/8+1, message2);

    printf("State after permutation:\n");
    printf("4F 7F 71 1E 80 D0 3E AB 9B 65 68 6A DB 1A 27 65 CE FC D0 A0 "
           "95 DE 98 D0 5E BE 6B 4F 9C 83 8F D4 6E 16 FC 52 C9 D9 7E D5 "
           "07 8C 0E 70 BB 2B 83 8A F9 FA DA 2A 26 3F 65 96 5C 1A 2D 65 "
           "FE DB 1A 9A 88 66 A9 44 F6 21 BB F6 D7 7E 7C EB 7F 3D CE 22 "
           "24 1D 1F 10 F0 A6 C8 CA 86 8D DC AF 0B 35 D0 8C 4D 35 7E 1A "
           "89 6D 43 44 28 21 DE 69 1E 53 FC 23 8D E6 83 04 6A 6F 15 CA "
           "0F 7A 89 AC 44 A1 90 E9 6F E7 FB 3A B8 94 BE 81 87 52 81 91 "
           "05 B7 0E 6B DA EC 39 B4 4C D9 EA E2 C2 9D 72 E7 F6 01 ED EF "
           "3D 83 F1 98 CE 99 5C B7 86 DA 99 F4 59 D6 5B 3E 85 8B FB AA "
           "44 81 E6 D7 9B D2 25 3E 02 E6 8D C0 2D 3D F0 D8 9D A6 57 B1\n"); 
    
    PRINT_GPU_RESULT;

    memset(h_data, 0x00, DATA_SIZE*sizeof(*h_data));
    memcpy(h_data, message3, 256); 
    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, DATA_SIZE/8*sizeof(*d_data)
                 , cudaMemcpyHostToDevice));

    keccac_kernel<<<1,32>>>(d_data, d_out, message3Length);

    CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));

    PRINT_CPU_INPUT((message3Length-1)/8+1, message3);

    printf("State after permutation:\n");
    printf("65 A3 44 32 73 7A F3 7E 99 62 23 B4 76 0F 80 2E 17 69 ED FF "
           "70 51 BD 84 3A C2 B4 82 C3 72 06 AC C8 5B 2A 58 C5 5C 4F 24 "
           "8D 0A C4 70 08 98 20 58 19 4C 51 7A E8 1E 9D 41 25 9C 7E 92 "
           "79 38 8B 3C 0B D6 B0 EF CA F0 D3 50 C3 D3 18 F0 53 15 C0 F8 "
           "B0 CC C0 25 52 74 C2 E7 82 E7 B7 C5 51 5F 68 95 62 25 FA 91 "
           "D5 75 08 21 51 D4 F2 07 C9 DB 80 80 AA 84 B7 C2 85 56 D8 2E "
           "54 A8 81 F0 3B E0 EC 26 C1 0B EC 2E D2 BD C9 46 20 2B DF CF "
           "A8 D6 A7 C1 61 BD C5 E8 C8 15 39 31 C7 86 9B 46 15 EC E4 96 "
           "86 6C 0F FF 36 35 4A BB 88 71 4D E5 89 14 B3 2A 2B 86 69 25 "
           "9A EF D3 81 A1 F0 20 96 05 60 69 BC E1 8E 59 91 43 7B 63 76\n"); 
    
    PRINT_GPU_RESULT;

    printf("--- Switching to squeezing phase for the last message ---\n\n");

    keccac_squeeze_kernel<<<1,32>>>(d_out);
    CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));

    printf("State after permutation:\n");
    printf("61 ED 25 07 69 84 18 FA F0 25 8C A7 AC CB 88 06 F5 3B 69 E1 "
           "80 A6 63 6E D8 D6 3D F7 B1 45 04 56 FF 9B E7 51 FE C0 3F 9B "
           "E0 59 1C AE D1 AF BC D3 19 32 1E CF BE D5 78 0E 28 F9 54 57 "
           "33 D5 61 43 6E 41 2F 19 E0 AA 68 EA 05 35 6C DB 43 B2 79 66 "
           "0A 0C C0 86 10 F4 91 10 48 A2 48 E5 F9 6A 28 85 93 BC 4D 5A "
           "52 E8 A3 72 29 D9 DC 50 99 98 9B EA B9 BF A9 D0 A7 BE C5 97 "
           "65 72 A8 67 A5 05 B6 38 5A 12 44 26 BD D2 88 6E 4D B0 CB 61 "
           "B9 33 36 BA 34 71 9C C7 A1 B0 F1 5C 68 4A 40 BB 07 77 E9 B4 "
           "33 0E 1F 62 F9 7D 87 ED C2 22 10 E2 59 31 BC 0E D6 24 F7 72 "
           "87 13 52 07 AB AE F5 05 FE DC 78 87 FE 16 4A 6B 6E 57 87 9C\n"); 

    PRINT_GPU_RESULT;

    keccac_squeeze_kernel<<<1,32>>>(d_out);
    CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));

    printf("State after permutation:\n");
    printf("47 0F FE 09 54 6D 5B 78 8F 46 C4 D0 5E 3E A0 C0 DE 94 09 B4 "
           "9E C5 22 83 C9 FE 78 3F BF 2A 67 6B F0 15 1D 21 15 89 CF CF "
           "EE 5B A9 82 93 CD 2A 43 03 64 23 F0 03 83 68 A6 56 88 49 43 "
           "20 53 16 20 26 03 9F 67 8F C9 AE 4E 96 2E 1E E0 61 53 50 67 "
           "5F 7B 00 3C 5A CA E0 D7 79 C9 64 9B 07 12 2A B1 0D BD DB FE "
           "71 C5 64 E9 55 99 15 BC D0 F7 DE 72 25 2C 05 44 0E E8 36 F6 "
           "A3 6D 76 7F 80 FD 02 55 4C D0 D6 D9 CE A2 2C 52 07 2A 86 7E "
           "41 1A 36 BF 3F 0B A5 9A 19 CE AD 52 E4 57 F4 42 82 41 C9 74 "
           "FA F7 DB 2E DD E0 18 10 67 7D 3F FD A7 4B 8D D0 D2 28 FD 4F "
           "88 33 33 9F 53 56 1F 13 08 94 39 43 CE E7 4D 11 C9 AA 67 E8\n");
 
    PRINT_GPU_RESULT;

    keccac_squeeze_kernel<<<1,32>>>(d_out);
    CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));

    printf("State after permutation:\n");
    printf("90 29 3E 22 9F BE 20 CC DA 4B DB 2F 4F 45 28 01 0A 3B 1F F7 "
           "59 B2 B0 DE 41 C6 45 2F 69 8F 9B 01 7B 5E 33 73 ED BE 22 A2 "
           "4C B5 DA 1E DA FC BF F5 69 4B 42 2F 38 C8 F4 D2 12 B8 98 D5 "
           "C3 74 45 54 04 5B 65 DC FE 39 4A 66 B6 22 B9 B4 27 FD AD EA "
           "CE B6 DC FC 73 F1 2A 8C 35 FE 23 1A 79 49 06 8A FA 2C 73 63 "
           "92 41 3D 5F 30 AD 2E 6B C7 37 8C 3A 6A BF 35 7C C9 C1 AF 04 "
           "A0 49 5E B1 9A E3 B1 5D A7 72 F2 20 F2 DE 1E 06 64 96 04 7B "
           "D9 E7 E5 B0 E0 CF 7B A4 8C B2 D8 6E 70 F2 12 5A 2D 8D 15 95 "
           "5E 9B 2A F1 80 EC 26 ED 66 57 59 46 56 8B D3 D6 F9 EE 3D FA "
           "01 B5 EE A2 73 02 80 70 A0 D6 7F F1 62 70 8E A2 F5 03 7D 45\n");

    PRINT_GPU_RESULT;

    CUDA_SAFE_CALL(cudaFree(d_data));
    CUDA_SAFE_CALL(cudaFreeHost(h_data));
    CUDA_SAFE_CALL(cudaFree(d_out));
    CUDA_SAFE_CALL(cudaFreeHost(h_out));
}


