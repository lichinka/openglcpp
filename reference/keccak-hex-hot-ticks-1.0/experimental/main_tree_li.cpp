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
#include "cuda_keccak_tree_li.cuh"

int main(int argc, const char** argv) {

    if(argc <= 3) {
        fprintf(stderr, "usage: <filename> <digestlength> <height 0-5>\n");
        exit(-1);
    }

    struct stat buf;
    char const *filename = argv[1];
    int d = (int)strtol(argv[2], (char **)NULL, 10);
    int H = (int)strtol(argv[3], (char **)NULL, 10);
 
    if(stat(filename, &buf) < 0) {
        fprintf(stderr, "stat %s failed: %s\n", filename, strerror(errno));
        return -1;
    } 

    if(buf.st_size == 0 || buf.st_size > MAX_FILE_SIZE) {
        fprintf(stderr, "%s wrong sized %d\n", filename, (int)buf.st_size);
        return -1;
    }

    if(H < 0 || H > 5) {
        fprintf(stderr, "wrong tree height %d\n", H);
        return -1;
    }

    if(errno != ERANGE && d > 0) {
        size_t const fsize = buf.st_size;;
        CUDA_SAFE_CALL(cudaSetDevice(0));
        call_keccak_tree_li_kernel(filename, d, H, fsize);
    }

    return (EXIT_SUCCESS);
}

/********************************** end-of-file ******************************/

