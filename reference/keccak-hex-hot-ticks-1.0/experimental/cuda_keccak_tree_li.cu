/*                                                                           */
/*       Keccak-f1600: Tree-based version. Mode: LI (leaf-interleaving)      */
/*                                                                           */
/* Author: Gerhard Hoffmann                                                  */
/*                                                                           */
/* Note: The implementation is could not be tested against a reference       */
/*       implementation. It is given without any guarantee. The tree         */
/*       growing mode is not implemented.                                    */
/*                                                                           */
/* NOTES:                                                                    */
/* As described in the main document, the input of the scheme are two binary */
/* strings: the prefix P and the input message M (main document, page 30).   */
/* Its input parameters [...] are the following:                             */
/*                                                                           */
/* -> the tree growing mode G = LI or G = FNG                                */
/* -> the height H of the tree      (0 <= H < 256)                           */
/* -> the degree D of the nodes     (0 <= D < 256)                           */
/* -> the leaf block size B [bits]. (B==0 mod 8, 1 <= B/8 < 2^16)            */
/* -> the prefix length |P| [bits]. (|P|==0 mode 8, 0 <= |P| < 256)          */
/*                                                                           */
/* The construction of the leaves is as follows (Algorithm 2, page 31):      */
/*                                                                           */
/* For each leaf L_j, 0 <= j <= L-1, set L_j to the empty string.            */
/* for i=0 to |M|-1 do                                                       */
/*     j = floor(i/B) mod L                                                  */
/*     Append bit i of M to L_j                                              */
/* end for                                                                   */
/*                                                                           */
/* Note that it does not say that a leaf can only load B bits of data.       */
/* Instead, the number B is used merely as a parameter for the interleave    */
/* technique. Hence, the user must be given the opportunity to choose the    */
/* height H of the tree. If H==0, then the choice is a rooted, trivial tree  */
/* with degree D==0.                                                         */
/*                                                                           */
/* Some profiling results for calling the kernels (leaf,internal(s),final):  */
/* ./keccak_tree_li <filename> 100 <height>                                  */
/*                                                                           */
/* File size [bytes] | H | Digest length [bits] |  Time [seconds]            */
/* ------------------+---+----------------------+----------------------      */
/*          53872640 | 0 |                 100  |       23.032263            */
/*                   | 1 |                      |        4.995800            */
/*                   | 2 |                      |        0.626488            */
/*                   | 3 |                      |        0.255480            */
/*                   | 4 |                      |        0.184566            */
/*                   | 5 |                      |        0.217418            */
/*                                                                           */
/* File size [bytes] | H | Digest length [bits] |  Time [seconds]            */
/* ------------------+---+----------------------+----------------------      */
/*           1287583 | 0 |                 100  |        0.550648            */
/*                   | 1 |                      |        0.120388            */
/*                   | 2 |                      |        0.016933            */
/*                   | 3 |                      |        0.008416            */
/*                   | 4 |                      |        0.013529            */
/*                   | 5 |                      |        0.054598            */
/*                                                                           */
/* As one can see, using the highest possible height of the tree does not    */
/* automatically mean the optimal time. The file is too small for such cases.*/
/**/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <cutil_inline.h>

#include <cuda_keccak_tree_li.cuh>

#define DIV         0x01
#define ROUNDS      24
#define R64(a,b,c)  (((a) << b) ^ ((a) >> c))

#define KECCAK_F1600(tr,ii) {\
    C[tr]    = A[s] ^ A[s+5] ^ A[s+10] ^ A[s+15] ^ A[s+20]; \
    D[tr]    = C[b[20+s]] ^ R64(C[b[5+s]],1,63);            \
    C[tr]    = R64(A[a[tr]]^D[b[tr]], ro[tr][0], ro[tr][1]);\
    A[d[tr]] = C[c[tr][0]] ^ ((~C[c[tr][1]]) & C[c[tr][2]]);\
    A[tr]   ^= rc[(tr==0) ? 0 : 1][ii];                     \
} 

#define KECCAK_F(tr) {\
    for(int i_,j_=0;j_<3;++j_) { \
        i_= 0+j_*8; KECCAK_F1600(tr,i_); i_= 1+j_*8; KECCAK_F1600(tr,i_);\
        i_= 2+j_*8; KECCAK_F1600(tr,i_); i_= 3+j_*8; KECCAK_F1600(tr,i_);\
        i_= 4+j_*8; KECCAK_F1600(tr,i_); i_= 5+j_*8; KECCAK_F1600(tr,i_);\
        i_= 6+j_*8; KECCAK_F1600(tr,i_); i_= 7+j_*8; KECCAK_F1600(tr,i_);\
    }\
}

#define PRINT_GPU_RESULT           \
    printf("\nOutput of GPU:\n");  \
    for(int i=0;i<200;++i) {       \
        printf("%02X ", h_out[i]); \
    } printf("\n\n\n"); 

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

#define SHA3_224 224
#define SHA3_256 256
#define SHA3_384 384
#define SHA3_512 512
#define SHA3_ARB 0

#define SHA3 SHA3_ARB

/* Following cases left open for now. */
#if SHA3==SHA3_224
/* SHA3-224: r = 1152, c = 448 -> truncate to 224 bits
*/
#elif SHA3==SHA3_256
/* SHA3-256: r = 1088, c = 512 -> truncate to 256 bits
*/
#elif SHA3==SHA3_384
/* SHA3-384: r = 832, c = 768  -> truncate to 384 bits
*/
#elif SHA3==SHA3_512
/* SHA3-512: r = 576, c = 1024 -> truncate to 512 bits
*/
#elif SHA3==SHA3_ARB
/* Default: arbitrary output length. r = 1024, c = 576
*/
#define BITRATE             1024
#define CAPACITY             576
#define PREFIX_LEN           ((uint8_t)(BITRATE-264)/8) 
#define INTERNAL_BSIZE_BYTES BITRATE
#define INTERNAL_BSIZE_BITS  (BITRATE<<3)
#define NODE_DEGREE          8
#define L0                   1
#define L1                   (L0*NODE_DEGREE)
#define L2                   (L1*NODE_DEGREE)
#define L3                   (L2*NODE_DEGREE)
#define L4                   (L3*NODE_DEGREE)
#define L5                   (L4*NODE_DEGREE)                   
#define BSIZE_LEAF           8192
#define BILLION              1000000000L

/* For all nodes, the scheme uses the sponge function defined by:            */
/*                                                                           */
/*****************************************************************************/
/* KeccakNS[r,c,ns](data) = Keccak[r,c,1](UTF-8(ns)||0x00||encode_{ns}(data) */
/*****************************************************************************/
/*                                                                           */

static uint8_t M_host[200] = {
    /* The first part of each message will be (UTF-8(ns)). As namespace name */
    /* we use ns := http://keccak.noekeon.org/tree/ , because UTF-8 is in    */
    /* this case the same as ASCII and because it is used in the main        */
    /* document as well.                                                     */     
    0x68, 0x74, 0x74, 0x70, 0x3a, 0x2f, 0x2f, 0x6b, 0x65, 0x63, 0x63, 
    0x61, 0x6b, 0x2e, 0x6e, 0x6f, 0x65, 0x6b, 0x65, 0x6f, 0x6e, 0x2e, 
    0x6f, 0x72, 0x67, 0x2f, 0x74, 0x72, 0x65, 0x65, 0x2f,
    /* concatenate 0x00                                                      */
    0x00,
    /* concatenate the length of the prefix (-> enc(|P|/8,8))                */
    PREFIX_LEN, 
    /* The prefix (key or salt) P, which is supposed to have 0 to 2040 bits. */
    /* In our case, it consists of 95 bytes for then M will have a length of */
    /* 128 bytes and will be aligned on a block boundary.                    */
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x90, 0x91, 0x92, 0x93, 0x94, 
    /* Fill the remaining 72 bytes with zeros. Then A[t] = PF[t]; can be     */
    /* by 25 threads in one step.                                            */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00 };
                                        /* Padding step for non-final nodes. */
static uint8_t P_host[200] = {
    0x01, /* pad(m)   */ 
    0x00, /* enc(0,8) */
    0x01,  DIV,  BITRATE/8, 0x01, /* Keccak padding */ 
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00
};

#endif

__device__ __constant__ uint64_t PF[25];
__device__ __constant__ uint64_t P[25];
__device__ __constant__ uint32_t a[25];
__device__ __constant__ uint32_t b[25];
__device__ __constant__ uint32_t c[25][3];
__device__ __constant__ uint32_t d[25];
__device__ __constant__ uint32_t ro[25][2];
__device__ __constant__ uint64_t rc[5][ROUNDS];

uint8_t *h_out;
uint64_t *d_out;
uint8_t *h_data;
uint64_t *d_data;
uint64_t *h_parent_data;
uint64_t *d_pdata;
uint64_t *d_ppdata;

__global__ /*                       Squeeze phase after absorbing all nodes. */
void keccac_squeeze_kernel(uint64_t *data) {/* In case a digest of length    */
    int const t = threadIdx.x;   /* greater than 1024 bit is needed, call    */           
    int const s = threadIdx.x%5; /*             this  kernel multiple times. */

    __shared__ uint64_t A[25];  
    __shared__ uint64_t C[25]; 
    __shared__ uint64_t D[25]; 

    if(t < 25) {
        A[t] = data[t];
        KECCAK_F(t);
        data[t] = A[t];
    }
}

__global__ /*        Processing a trivial tree containing only a final node. */
void keccak_root(uint64_t *data, uint64_t *out, uint64_t databitlen) {
    
    int const t = threadIdx.x; 

    __shared__ uint64_t A[25];  
    __shared__ uint64_t B[25];  
    __shared__ uint64_t C[25]; 
    __shared__ uint64_t D[25]; 

    if(t < 25) {

        int const s = threadIdx.x%5;
        int const blocks = databitlen/BITRATE;

        /* For each node, first load the predefined header. As pointed out   */ 
        /* on page 29 of the main document, it could be precomputed as it is */
        /* the same for each node.                                           */
        A[t] = PF[t]; KECCAK_F(t);
        
        B[t] = 0ULL;
        if(t < 16) 
            B[t] = data[t]; 

        for(int block=0;block<blocks;++block) { 
    
            A[t] ^= B[t];
            data += BITRATE/64;

            if(t < 16) 
                B[t] = data[t];                             /* prefetch data */
            KECCAK_F(t);

            databitlen -= BITRATE;
        }
        /*                   Note that there are already data loaded into B. */
        int const bytes = databitlen/8;  /* 'bytes' will be smaller than     */
                                         /* BITRATE/8, so the final padding  */
        if(t == 0) {                     /* will not overflow B.             */
            uint8_t *p = (uint8_t *)B+bytes; 
            uint8_t const q = *p;
            /* The padding pad(m) first shifts the bits left down in         */
            /* direction LSB and then 'or's the result with a one shifted to */
            /* the left. So the result of the operation for the byte         */
            /* [11100000] (where databitlen is 3) is [00001111]. Note that   */
            /* if databitlen equals 0, then we pad 0x01 as is should be.     */
            *p++ = (q >> (8-(databitlen&7)) | (1 << (databitlen&7)));
            *p++ = 0x00;        /* Pad enc(0,8) (because we are in LI mode)  */ 
            *p++ = 0x00;        /* enc(H,8), where H==0                      */
            *p++ = 0x00;        /* enc(D,8), where D==0                      */
            *(uint16_t *)p++ = BSIZE_LEAF/8; /* enc(bsize_leaf/8,16)         */
            /* Finally, we concatenate the paddings left by Keccak itself.   */
            *p++ = 0x01; *p++ = DIV; *p++ = BITRATE/8; *p++ = 0x01;
            while(p < (uint8_t *)&B[25])
                *p++ = 0;
        }
        A[t] ^= B[t]; KECCAK_F(t);       /* Add to state and call Keccak-f   */

        if((bytes+4) > BITRATE/8) {      /* thread 0 has crossed the 128 byte*/
            if(t < 16) B[t] = 0ULL;      /* boundary and touched some higher */
            if(t <  9) B[t] = B[t+16];   /* parts of B.                      */
            if(t < 16) A[t] ^= B[t];
           
            KECCAK_F(t); 
        } 
        /* All 25 threads write the result, because there is no truncation   */
        /* for final nodes. Potential squeeze calls are done in a separate   */
        /* kernel.                                                           */
        out[t] = A[t];
    }
}

__global__ /*                Processing the final node of a non-trivial tree */
void keccak_final(uint64_t *data, uint64_t *out, uint8_t H) {

    int const tid = threadIdx.x;      /* local thread id == global thread id */
    int const t = tid%32;             /* thread number local to the warp     */

    __shared__ uint64_t A[25];        /* state memory */
    __shared__ uint64_t B[25]; 
    __shared__ uint64_t C[25]; 
    __shared__ uint64_t D[25]; 

    if(t < 25) {
        int const s = t%5;

        /* For each node, first load the predefined header. As pointed out   */ 
        /* on page 29 of the main document, it could be precomputed as it is */
        /* the same for each node.                                           */
        A[t] = PF[t]; KECCAK_F(t);

        B[t] = 0ULL;

        for(int j=0;j<NODE_DEGREE;++j) { 
            if(t < 16) 
                B[t] = data[t];    /* fetch data and call Keccak permutation */
            A[t] ^= B[t];
            KECCAK_F(t);
            data += BITRATE/64;
        }

        /* Because we are in the final node of a non-trivial tree, all that  */ 
        /* remains to be done is to pad the empty message with 0x01 and      */
        /* enc(0,8) = 0, add the padding of Keccak itself and call Keccak-f. */
        B[t] = 0ULL; 

        if(t == 0) {
            uint8_t *__restrict__ p = (uint8_t *)B; 
            *p++ = 0x01;        /* pad(m)                                    */
            *p++ = 0x00;        /* Pad enc(0,8) (because we are in LI mode)  */ 
            *p++ = H;           /* enc(H,8)                                  */
            *p++ = NODE_DEGREE; /* enc(D,8)                                  */
            *(uint16_t *)p++ = BSIZE_LEAF/8; /* enc(bsize_leaf/8,16)         */
            /* Finally, we concatenate the paddings left by Keccak itself.   */
            *p++ = 0x01; *p++ = DIV; *p++ = BITRATE/8; *p++ = 0x01;
        }
        /* All 25 threads write the result, because there is no truncation   */
        /* for final nodes. Potential squeeze calls are done in a separate   */
        /* kernel.                                                           */
        A[t] ^= B[t]; KECCAK_F(t);         /* Add to state and call Keccak-f */
                   /* Potential squeeze calls are done in a separate kernel. */
        out[t] = A[t];
    }
}

/* Processing of an internal node except the final node. Unlike with leaf    */
/* nodes, input data are always bsize_leaf bits in size and are always       */
/* aligned.                                                                  */
template<uint32_t LEAVES>
__global__
void keccak_internal(uint64_t *d_data, uint64_t *pdata) {
    int const tid = threadIdx.x;                          /* local thread id */
    int const bid = blockIdx.x + blockIdx.y*gridDim.x;    /* global block id */
    int const gtid = tid + blockDim.x*bid;                     /* global tid */
    int const leaf_id = gtid/32;                   /* global leaf (=warp) id */

    int const tw = tid/32;          /* warp of the thread local to the block */
    int const tr = tid%32;          /* thread number local to the warp       */

    __shared__ uint64_t A_[8][25]; /* state memory */
    __shared__ uint64_t C_[8][25]; 
    __shared__ uint64_t D_[8][25]; 

    if(tr < 25) {
        int const s  = tr%5;
        /*each thread sets a pointer to its corresponding leaf (=warp) memory*/
        uint64_t *__restrict__ A = &A_[tw][0]; 
        uint64_t *__restrict__ C = &C_[tw][0], *__restrict__ D = &D_[tw][0];

        /* For each node, first load the predefined header. As pointed out   */ 
        /* on page 29 of the main document, it could be precomputed as it is */
        /* the same for each node.                                           */
        A[tr] = PF[tr]; KECCAK_F(tr);

        uint64_t *__restrict__ data = &d_data[(BSIZE_LEAF*leaf_id)/64];

        for(int j=0;j<NODE_DEGREE;++j) { 
            C[tr] = 0ULL;             /* LOOP ASSUMPTION:                    */
            if(tr < 16)               /* node_degree * bitrate == bsize_leaf */
                C[tr] = data[tr];     /* fetch data and call Keccak          */
            A[tr] ^= C[tr];
            KECCAK_F(tr);
            data += BITRATE/64;
        }

        A[tr] ^= P[tr]; KECCAK_F(tr);         /* Add to state, call Keccak-f */

        /* Each leaf generates bitrate/8 bytes of data for the next level.   */
        if(tr < 16) { /* 16 threads of each leaf copy the data to the parent.*/
            uint64_t (*par)[LEAVES][16] = (uint64_t (*)[LEAVES][16])(pdata);
            (*par)[leaf_id][tr] = A[tr]; 
        }
    }        
}

/* The first step is to interpolate (=interleaf) the data to the leaves of   */
/* tree (see algorithm 2 in the main document, p. 31).                       */
template<uint32_t LEAVES>
__global__
void keccak_leaf(uint64_t const *d_data, uint64_t *pdata, uint64_t databitlen) {
    int const tid = threadIdx.x;                          /* local thread id */
    int const bid = blockIdx.x + blockIdx.y*gridDim.x;    /* global block id */
    int const gtid = tid + blockDim.x*bid;                     /* global tid */
    int const leaf_id = gtid/32;                   /* global leaf (=warp) id */

    int const tw = tid/32;          /* warp of the thread local to the block */
    int const tr = tid%32;          /* thread number local to the warp       */

    __shared__ uint64_t A_[8][25]; /* state memory */
    __shared__ uint64_t B_[8][25]; 
    __shared__ uint64_t C_[8][25]; 
    __shared__ uint64_t D_[8][25]; 

    if(tr < 25) {
        int const s = tr%5;
        /*each thread sets a pointer to its corresponding leaf (=warp) memory*/
        uint64_t *__restrict__ A = &A_[tw][0], *__restrict__ B = &B_[tw][0]; 
        uint64_t *__restrict__ C = &C_[tw][0], *__restrict__ D = &D_[tw][0];

        /* For each node, first load the predefined header. As pointed out   */ 
        /* on page 29 of the main document, it could be precomputed as it is */
        /* the same for each node.                                           */
        A[tr] = PF[tr]; KECCAK_F(tr);
        B[tr] = 0ULL;
       
        /* The next step to be done is now M = M||pad(m,8), see Algorithm 1. */ 
        /* Before the actual padding, we have to load the data using the     */
        /* interleaved fashion.                                              */

        /* First we check if each leaf could load as many as block size bits */
        /* of data. The leaf block size is denoted as B in the main document.*/
        /* Here it is denoted by bsize_leaf.                                 */
 
        /* A full stripe means that all the leaves can load bsize_leaf bits  */ 
        int const stripe_size = LEAVES*BSIZE_LEAF;               /* of data. */
        int const stripes = databitlen/stripe_size;
        int stripe = 0;
        uint64_t const *__restrict__ data;

        if(stripes > 0) { 

            data = &d_data[(BSIZE_LEAF*leaf_id)/64];

            for(stripe=0;stripe<stripes;++stripe) {/*     all leaves load    */
                                               /* bsize_leaf bits of data.   */
                if(tr < 16) 
                    B[tr] = data[tr];          /* fetch data and call Keccak */

                for(int j=0;j<NODE_DEGREE;++j) {
                    A[tr] ^= B[tr];     /* LOOP ASSUMPTION:                  */
                    data += BITRATE/64; /* node_degree*bitrate == bsize_leaf */
                    if(tr < 16)
                        B[tr] = data[tr];                   /* prefetch data */
                    KECCAK_F(tr);
                } 
    
                databitlen -= stripe_size;
            }
        }
        /* Secondly, we check if some leaves still can load bsize_leaf bits  */
        /* of data. Note that this can happen only once per leaf. If the     */
        /* data have been used, then only some padding remains to be done    */
        /* for this leaf.                                                    */
        if(leaf_id < databitlen/BSIZE_LEAF) {/* databitlen might have been   */
                                             /* changed in the previous step */
            data = &d_data[(stripe*stripe_size+BSIZE_LEAF*leaf_id)/64];

            if(tr < 16) 
                B[tr] = data[tr];              /* fetch data and call Keccak */
            
            for(int j=0;j<NODE_DEGREE;++j) {
                A[tr] ^= B[tr];       /* LOOP ASSUMPTION:                    */
                data += BITRATE/64;   /* node_degree * bitrate == bsize_leaf */
                if(tr < 16)
                    B[tr] = data[tr]; /* prefetch data */
                KECCAK_F(tr);
            
                databitlen -= BITRATE;
            } 

            /* The data chunk has been aligned at a byte boundary, and this  */
            /* node is not a final node. What remains to be done according to*/
            /* Algorithm 1 is M||pad(m) and M=M||enc(0,8).                   */
            /* Finally, we have to do the last four paddings of Keccak in-   */
            /* volving the diversifier and the bitrate/8.                    */
            /* For this purpose, we use the predefined P-array.              */

            A[tr] ^= P[tr]; KECCAK_F(tr); /* Add to the state, call Keccak-f */

        } else 
        /* The leaf cannot load bsize_leaf bits. Instead, it loads in a first*/
        /* step bitrate-sized chunks of bits. After that there still might be*/
        /* data which is not aligned on a bitrate-sized boundary. Therefore, */
        /* to align it on a bitrate boundary, some padding has to be done.   */
        if(leaf_id*BSIZE_LEAF < databitlen) {

            data = &d_data[(stripe*stripe_size+BSIZE_LEAF*leaf_id)/64];
            databitlen = databitlen%BSIZE_LEAF;    

            int bitrate_blocks = databitlen/BITRATE;    

            if(tr < 16) 
                B[tr] = data[tr];              /* fetch data and call Keccak */
            
            for(int j=0;j<bitrate_blocks;++j) { 
                A[tr] ^= B[tr];       /* LOOP ASSUMPTION:                    */
                data += BITRATE/64;   /* node_degree * bitrate == bsize_leaf */
                if(tr < 16)
                    B[tr] = data[tr]; /* prefetch data */
                KECCAK_F(tr);

                databitlen -= BITRATE;
            }
            /*               Note that there are already data loaded into B. */
            int const bytes = databitlen/8;  /* 'bytes' will be smaller than */
                                         /* BITRATE/8, so the final padding  */
            if(tr == 0) {                /* will not overflow B.             */
                uint8_t *p = (uint8_t *)B+bytes; 
                uint8_t const q = *p;
                /* The padding pad(m) first shifts the bits left down in     */
                /* direction LSB and then 'or's the result with a one shifted*/
                /* to the left. So the result of the operation for the byte  */
                /* [11100000] (where databitlen is 3) is [00001111]. Note    */
                /* that if databitlen equals 0, we pad 0x01 as is should be. */
                *p++ = (q >> (8-(databitlen&7)) | (1 << (databitlen&7)));
                *p++ = 0x00;        /* Pad enc(0,8) (because of LI mode)     */ 
                /* Finally, concatenate the paddings left by Keccak itself.  */
                *p++ = 0x01; *p++ = DIV; *p++ = BITRATE/8; *p++ = 0x01; 
                while(p < (uint8_t *)&B[25])
                    *p++ = 0;
            }

            A[tr] ^= B[tr]; KECCAK_F(tr); /* Add to state and call Keccak-f  */

            if((bytes+4) > BITRATE/8) { /* thread 0 has crossed the 128 byte */
                if(tr < 16) B[tr] = 0ULL;/* boundary and touched some higher */
                if(tr <  9) B[tr] = B[tr+16]; /* parts of B.                 */
                if(tr < 16) A[tr] ^= B[tr];
           
                KECCAK_F(tr); 
            } 

        } else {/* Leaves of this kind are called upon an empty message m.   */
            /* All that remains to be done is to pad the empty message with  */
            /* 0x01 and enc(0,8) = 0x00, add the padding of Keccak itself and*/
            /* call Keccak-f.                                                */
            A[tr] ^= P[tr]; KECCAK_F(tr); /* add to the state, call Keccak-f */
        }
        /* Each leaf generates bitrate/8 bytes of data for the next level.   */
        /* 16 threads of each leaf copy the data to the parent.              */
        if(tr < 16) {
            uint64_t (*par)[LEAVES][16] = (uint64_t (*)[LEAVES][16])(pdata);
            (*par)[leaf_id][tr] = A[tr]; 
        }
    }
}

void call_keccak_tree_li_kernel(char const *filename
                              , uint32_t digestlength
                              , uint32_t H
                              , uint32_t fsize) {
    struct timespec start, stop;
    double accum;
                                        /* align the data on BITRATE/8 bytes */
    uint32_t size = ((fsize-1)/(BITRATE/8) + 1)*(BITRATE/8);

    /* initialize the memory */
    h_out  = (uint8_t *)malloc(200);
    h_data = (uint8_t *)calloc(1, size);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, size));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pdata, size/L1));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_ppdata, size/L2));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_out, 25*sizeof(uint64_t)));

    FILE *in = fopen(filename, "r");           /* read the file to be hashed */
    if(in == NULL) {
        fprintf(stderr, "open %s failed: %s\n", filename, strerror(errno));
        return;
    }
    if(fread(h_data, 1, fsize, in) < fsize) { 
        fprintf(stderr, "read %s failed: %s\n", filename, strerror(errno));
        return;
    }
    fclose(in);

    /* copy the constant data to the GPU */ 
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(PF, M_host, sizeof(M_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(P , P_host, sizeof(P_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(a , a_host, sizeof(a_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(b , b_host, sizeof(b_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c , c_host, sizeof(c_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d , d_host, sizeof(d_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ro, rho_offsets, sizeof(rho_offsets)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rc, round_const, sizeof(round_const)));

    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    int count = 0;
    for(int i=0;i<8;++i) {
        if((h_data[fsize-1] >> i) & 1) {                /* compute bit count */ 
            count = 8 - i; break;
        }
    }

    dim3 grid5(64, 64, 1),           /* 2-D Grid layout: 64x64 = 4096 blocks */
         grid4(32, 16, 1), 
         grid3( 8,  8, 1), 
         grid2( 8,  1, 1), 
         grid1( 1,  1, 1);

    if(clock_gettime(CLOCK_REALTIME, &start) == -1) {
        perror("clock gettime");
        return;
    }

    int const bitcount = (fsize-1)*8 + count;

    /* Essentially Algorithm 3. The special structure of the GPU allows an   */
    /* iterative calling sequence, because global memory on the GPU remains  */
    /* its values across kernel calls.                                       */
    if(H == 0) { /*      1 leave */
        keccak_root        <<<1,32>>>(d_data, d_out, bitcount);
    } else
    if(H == 1) { /*     8 leaves */
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_pdata, L1*16*8));
        keccak_leaf    <L1><<<grid1, 256>>>(d_data , d_pdata, bitcount);
        keccak_final       <<<grid1,  32>>>(d_pdata, d_out, H);
    } else
    if(H == 2) { /*    64 leaves */
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_pdata , L2*16*8));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_ppdata, L1*16*8));
        keccak_leaf    <L2><<<grid2, 256>>>(d_data  , d_pdata, bitcount);
        keccak_internal<L1><<<grid1, 256>>>(d_pdata , d_ppdata);
        keccak_final       <<<grid1,  32>>>(d_ppdata, d_out, H);
    } else
    if(H == 3) { /*   512 leaves */
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_pdata , L3*16*8));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_ppdata, L2*16*8));
        keccak_leaf    <L3><<<grid3, 256>>>(d_data  , d_pdata, bitcount);
        keccak_internal<L2><<<grid2, 256>>>(d_pdata , d_ppdata);
        keccak_internal<L1><<<grid1, 256>>>(d_ppdata, d_pdata);
        keccak_final       <<<grid1,  32>>>(d_pdata , d_out, H);
    } else
    if(H == 4) { /*  4096 leaves */
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_pdata , L4*16*8));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_ppdata, L3*16*8));
        keccak_leaf    <L4><<<grid4, 256>>>(d_data  , d_pdata, bitcount);
        keccak_internal<L3><<<grid3, 256>>>(d_pdata , d_ppdata);
        keccak_internal<L2><<<grid2, 256>>>(d_ppdata, d_pdata);
        keccak_internal<L1><<<grid1, 256>>>(d_pdata , d_ppdata);
        keccak_final       <<<grid1,  32>>>(d_ppdata, d_out, H);
    } else
    if(H == 5) { /* 32768 leaves */
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_pdata , L5*16*8));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_ppdata, L4*16*8));
        keccak_leaf    <L5><<<grid5, 256>>>(d_data  , d_pdata, bitcount);
        keccak_internal<L4><<<grid4, 256>>>(d_pdata , d_ppdata);
        keccak_internal<L3><<<grid3, 256>>>(d_ppdata, d_pdata);
        keccak_internal<L2><<<grid2, 256>>>(d_pdata , d_ppdata);
        keccak_internal<L1><<<grid1, 256>>>(d_ppdata, d_pdata);
        keccak_final       <<<grid1,  32>>>(d_pdata , d_out, H);
    } 

    /*  Blocks until the device has completed all preceding requested tasks. */
    cudaThreadSynchronize();

    if(clock_gettime(CLOCK_REALTIME, &stop) == -1) {
        perror("clock gettime");
        return;
    }

    accum = (stop.tv_sec - start.tv_sec) 
          + (double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;
    printf("\n--- Profiling results ---\n");
    printf("File size (aligned to 128 bytes): %ld\n", size);
    printf("Runtime of kernel calls: %lf [s]\n", accum);

    CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));
    PRINT_GPU_RESULT;

    for(int i=0;i<digestlength/BITRATE;++i) {
        keccac_squeeze_kernel<<<1,32>>>(d_out);
        CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));
        PRINT_GPU_RESULT;
    }

    free(h_out);
    free(h_data);
    CUDA_SAFE_CALL(cudaFree(d_out));
    CUDA_SAFE_CALL(cudaFree(d_data));

    if(H > 0) {
        CUDA_SAFE_CALL(cudaFree(d_ppdata));
        CUDA_SAFE_CALL(cudaFree(d_pdata));
    }
}
/********************************** end-of-file ******************************/

