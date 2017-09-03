
/* Author: Gerhard Hoffmann                                                  */
/*                                                                           */
/* Batch implementation of Keccak-f1600. Calculates for 960 files as input as*/
/* output: the corresonding digest (of theoretical infinite length ->        */
/* bitrate r = 1024).                                                        */
/*                                                                           */
/* Note: The code below shows how to proceed in principle. It hashes the     */
/* same file 960 times (with of course the same result). For a more realistic*/
/* scenario it would be necessary to have something like an input file       */
/* containing the filenames to be hashed. Another point would be to make sure*/
/* that the GTX 295 with only ~2x900 MB main memory can handle all the files.*/
/* An approach would be to use streams. Another one to come up with kind of  */
/* a scheduling algorithm for hashing what files first if the sizes of the   */
/* files show a high variation.                                              */
/*                                                                           */
/* The kernels below have been executed on a grid with 120 blocks, where     */
/* each block consists of 256 threads. The occupancy was reported as 0.5 by  */
/* the CUDA profiler. The kernels have been based straight on the kernels of */
/* the basic implementation.                                                 */
/*                                                                           */
/* Note that the GTX 295 has two built in GPUs. For this implementation only */
/* one of them has been used. Using both GPUs it would be necessary to use   */
/* either two different CPU processes or CPU threads. Because the GPU scales */
/* almost perfectly, we considered it as not essential to use both cards.    */
/*                                                                           */
/* Some values reported by the CUDA profiler (hashing the same file 960x).   */
/*                                                                           */
/* File size [bytes] | Digest length [bits] |  Time [nanoseconds]            */
/* ------------------+----------------------+----------------------          */
/*              3510 |                 100  |           11244544             */
/*             13227 |                 100  |           41634434             */
/*             42653 |                 100  |          133587688             */
/*            110455 |                 100  |          345085750             */
/**/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <errno.h>
#include <cutil_inline.h>
#include <cuda_keccak_batch.cuh>


#define PRINT_GPU_RESULT           \
    printf("\nOutput of GPU:\n");  \
    for(int i=0;i<200;++i) {       \
        printf("%02X ", h_out[j][i]); \
    } printf("\n\n\n"); 

#define BLOCK_SIZE    256 /* threads per block */
#define BLOCKS_PER_SM 4   /* threads per streaming multiprocessor */
#define WARPS_PER_SM  ((BLOCK_SIZE/32)*BLOCKS_PER_SM)
#define SM            30
#define FILES         (WARPS_PER_SM*SM)
#define MAX_FILE_SIZE 1000000000
#define BITRATE       1024
#define ROUNDS        24 
#define OFFSET        63 
#define R64(a,b,c) (((a) << b) ^ ((a) >> c)) /* works on the GPU also for 
                                                b = 64 or c = 64 */
static uint8_t  **h_data;
static uint8_t  **h_out;
static uint64_t **d_data;
static uint64_t **d_out;
static uint64_t *h_dblen;
static uint64_t *d_dblen;
static uint64_t *d_data2[FILES];
static uint64_t *d_out2[FILES];

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
void keccac_squeeze_kernel(uint64_t **data) {/* In case a digest of length  */
                                 /* greater than 1024 bits is needed, call  */
    int const tid = threadIdx.x; /* this kernel multiple times. Another way */
    int const tw  = tid/32;      /* would be to have a loop here and squeeze*/
    int const t   = tid%32;      /* more than once.                         */
    int const s   = t%5;
    int const gw  = (tid + blockIdx.x*blockDim.x)/32; 

    __shared__ uint64_t A_[8][25];  
    __shared__ uint64_t C_[8][25]; 
    __shared__ uint64_t D_[8][25]; 

    if(t < 25) {
        /*each thread sets a pointer to its corresponding leaf (=warp) memory*/
        uint64_t *__restrict__ A = &A_[tw][0];
        uint64_t *__restrict__ C = &C_[tw][0]; 
        uint64_t *__restrict__ D = &D_[tw][0]; 

        A[t] = data[gw][t];

        for(int i=0;i<ROUNDS;++i) {                              /* Keccak-f */
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }

        data[gw][t] = A[t];
    }
}
/* The batch kernel is executed in blocks consisting of 256 threads. The     */
/* basic implementation of Keccak uses only one warp of 32 threads. Therefore*/
/* the batch kernel executes 8 such warps in parallel.                       */
__global__
void keccac_kernel(uint64_t **d_data, uint64_t **out, uint64_t *dblen) {

    int const tid = threadIdx.x; 
    int const tw  = tid/32;         /* warp of the thread local to the block */
    int const t   = tid%32;         /* thread number local to the warp       */
    int const s   = t%5;
    int const gw  = (tid + blockIdx.x*blockDim.x)/32; /* global warp number  */

    __shared__ uint64_t A_[8][25];  /* 8 warps per block are executing Keccak*/ 
    __shared__ uint64_t B_[8][25];  /*  in parallel.                         */
    __shared__ uint64_t C_[8][25]; 
    __shared__ uint64_t D_[8][25];

    if(t < 25) {/* only the lower 25 threads per warp are active. each thread*/
                /* sets a pointer to its corresponding warp memory. This way,*/
                /* no synchronization between the threads of the block is    */
                /* needed. Threads in a warp are always synchronized.        */
        uint64_t *__restrict__ A = &A_[tw][0], *__restrict__ B = &B_[tw][0]; 
        uint64_t *__restrict__ C = &C_[tw][0], *__restrict__ D = &D_[tw][0];
        uint64_t *__restrict__ data = d_data[gw];

        uint64_t databitlen = dblen[gw];
        
        A[t] = 0ULL;
        B[t] = 0ULL;
        if(t < 16) B[t] = data[t]; 

        int const blocks = databitlen/BITRATE;
       
        for(int block=0;block<blocks;++block) {/* load data without crossing */
                                                     /* a 128-byte boundary. */                
            A[t] ^= B[t];

            data += BITRATE/64;
            if(t < 16) B[t] = data[t];                      /* prefetch data */

            for(int i=0;i<ROUNDS;++i) {                          /* Keccak-f */
                C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
                D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
                C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
                A[t] ^= rc[(t==0) ? 0 : 1][i]; 
            }

            databitlen -= BITRATE;
        }

        int const bytes = databitlen/8;

        if(t == 0) {                              /* pad the end of the data */
            uint8_t *p = (uint8_t *)B+bytes;
            uint8_t const q = *p;
            *p++ = (q >> (8-(databitlen&7)) | (1 << (databitlen&7)));
            *p++ = 0x00; 
            *p++ = BITRATE/8; 
            *p++ = 0x01; 
            while(p < (uint8_t *)&B[25])
                *p++ = 0;
        }
        if(t < 16) A[t] ^= B[t];                    /* load 128 byte of data */
        
        for(int i=0;i<ROUNDS;++i) {                              /* Keccak-f */
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }

        if((bytes+4) > BITRATE/8) {/*then thread t=0 has crossed the 128 byte*/
            if(t < 16) B[t] = 0ULL;/* boundary and touched some higher parts */
            if(t <  9) B[t] = B[t+16]; /* of B.                              */
            if(t < 16) A[t] ^= B[t];
            
            for(int i=0;i<ROUNDS;++i) {                          /* Keccak-f */
                C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
                D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
                C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
                A[t] ^= rc[(t==0) ? 0 : 1][i]; 
            }
        } 

        out[gw][t] = A[t]; /* write the result */
    }
}
/**/
/**/
/**/
void call_keccak_batch_kernel(char const *filename, int digestlength) {

    struct stat buf;
    size_t size;
 
    if(stat(filename, &buf) < 0) {
        fprintf(stderr, "stat %s failed: %s\n", filename, strerror(errno));
        return;
    } 

    if(buf.st_size == 0 || buf.st_size > MAX_FILE_SIZE/FILES) {
        fprintf(stderr, "%s wrong sized %d\n", filename, (int)buf.st_size);
        return;
    }
                                        /* align the data on BITRATE/8 bytes */
    size = ((buf.st_size-1)/(BITRATE/8) + 1)*(BITRATE/8);

    h_data  = (uint8_t **)malloc(FILES*sizeof(*h_data));
    h_out   = (uint8_t **)malloc(FILES*sizeof(*h_out));
    h_dblen = (uint64_t *)malloc(FILES*sizeof(*h_dblen));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_dblen, FILES*sizeof(*d_dblen)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, FILES*sizeof(*d_data)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_out, FILES*sizeof(*d_out)));

    for(int i=0;i<FILES;++i) {             /* allocate memory for each file */
        h_data[i] = (uint8_t *)malloc(size);  /* and for each output buffer */
        h_out[i] = (uint8_t *)malloc(200);
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_data2[i], size));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_out2[i], 200));
    }

    CUDA_SAFE_CALL(cudaMemcpy(d_data, d_data2    /* copy the device pointers */
        , FILES*sizeof(d_data2[0]), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_out, d_out2
        , FILES*sizeof(d_out2[0]), cudaMemcpyHostToDevice));

    FILE *in = fopen(filename, "r");

    if(in == NULL) {
        fprintf(stderr, "open %s failed: %s\n", filename, strerror(errno));
        return;
    }

    memset(&h_data[0][0], 0x00, size);                   /* read the file(s) */ 
    if(fread(&h_data[0][0], 1, (size_t)buf.st_size, in) < buf.st_size) {
        fprintf(stderr, "read %s failed: %s\n", filename, strerror(errno));
        return;
    }
    for(int i=1;i<FILES;++i) { /* copy the file content (only for this test) */
        memcpy(h_data[i], h_data[0], size);  
    }

    fclose(in);

    for(int j=0;j<FILES;++j) { 
        int count = 0;
        for(int i=0;i<8;++i) {
            if((h_data[j][buf.st_size-1] >> i) & 1) {   /* compute bit count */ 
                count = 8 - i; break;
            }
        }
        h_dblen[j] = (buf.st_size-1)*8 + count;
    }
    CUDA_SAFE_CALL(cudaMemcpy(d_dblen, h_dblen, FILES*sizeof(*h_dblen)
                 , cudaMemcpyHostToDevice));
                                  /* copy the Keccak tables from host to GPU */
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(a, a_host, sizeof(a_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(b, b_host, sizeof(b_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c, c_host, sizeof(c_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d, d_host, sizeof(d_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ro, rho_offsets, sizeof(rho_offsets)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rc, round_const, sizeof(round_const)));

    for(int i=0;i<FILES;++i) {          /* copy the file contents to the GPU */
        CUDA_SAFE_CALL(cudaMemcpy(d_data2[i], h_data[i], size
                     , cudaMemcpyHostToDevice));
    }
    /* call the GPU */
    keccac_kernel<<<BLOCKS_PER_SM*SM,BLOCK_SIZE>>>/*BLOCKS_PER_SM*SM==FILES/8*/
        (d_data, d_out, d_dblen);
    
    for(int j=0;j<2/*FILES*/;++j) { /* fetch only two of the hashed files to */
        memset(h_out[j], 0x00, 200);                /* check for correctness */
        CUDA_SAFE_CALL(cudaMemcpy(h_out[j], d_out2[j], 200
                     , cudaMemcpyDeviceToHost));
        printf("FILE %03d:", j);
        PRINT_GPU_RESULT;
    }

    for(int j=0;j<digestlength/BITRATE;++j) { /* GPU: call the squeeze phase */
        keccac_squeeze_kernel<<<BLOCKS_PER_SM*SM, BLOCK_SIZE>>>(d_out);
        CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));
        PRINT_GPU_RESULT;
    }

    for(int i=0;i<FILES;++i) {                             /* release memory */
        CUDA_SAFE_CALL(cudaFree(d_data2[i]));
        CUDA_SAFE_CALL(cudaFree(d_out2[i]));
        free(h_out[i]);
        free(h_data[i]);
    }
    CUDA_SAFE_CALL(cudaFree(d_data));
    CUDA_SAFE_CALL(cudaFree(d_out));
    CUDA_SAFE_CALL(cudaFree(d_dblen));
    free(h_dblen);
    free(h_data);
    free(h_out);
}
/********************************* end-of-file *******************************/

