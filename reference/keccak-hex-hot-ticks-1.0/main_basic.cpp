/*
    Author: Gerhard Hoffmann
    Basic implementation of Keccak on the GPU (NVIDIA)
    Model used: GTX 295
*/

#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <ctype.h>
#include <errno.h>

#include <shrUtils.h>
#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include "cuda_keccak_basic.cuh"

int main(int argc, const char** argv) {

    if(argc <= 2) {
        fprintf(stderr, "usage: <filename> <digestlength>\n");
        exit(-1);
    }

    char const *filename = argv[1];
    int d = (int)strtol(argv[2], (char **)NULL, 10);

    if(errno != ERANGE && d > 0) {
        CUDA_SAFE_CALL(cudaSetDevice(0));
        call_keccak_basic_kernel(filename, d);
    }

    return (EXIT_SUCCESS);
}

/********************************** end-of-file ******************************/

