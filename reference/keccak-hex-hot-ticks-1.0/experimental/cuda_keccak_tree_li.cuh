#ifndef CUDA_KECCAK_TREE_LI_CUH_INCLUDED
#define CUDA_KECCAK_TREE_LI_CUH_INCLUDED

#include <inttypes.h>

#define MAX_FILE_SIZE 800000000

extern void call_keccak_tree_li_kernel(char const* filename
                                     , uint32_t digestlength
                                     , uint32_t H
                                     , uint32_t size);

#endif /* CUDA_KECCAK_TREE_LI_CUH_INCLUDED */

