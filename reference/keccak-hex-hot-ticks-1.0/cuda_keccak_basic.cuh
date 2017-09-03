#ifndef CUDA_KECCAK_BASIC_CUH_INCLUDED
#define CUDA_KECCAK_BASIC_CUH_INCLUDED

#include <inttypes.h>

extern void call_keccak_basic_kernel(char const *filename, int digestlength);

#endif /* CUDA_KECCAK_BASIC_CUH_INCLUDED */

