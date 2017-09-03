#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <ctype.h>
#include <errno.h>

// utilities and system includes
#include <shrUtils.h>

// CUDA-C includes
#include <cutil_inline.h>
#include <cuda_runtime_api.h>

#include "cuda_keccak.cuh"
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

static const char* help = 
"keccak [OPTIONS] file1 file2 ...\n"
"Options:\n"
"\t-h\tPrint this help string\n"
"\t-dnnnn\tSet digest length d to nnnn bits, 1 <= d <= 8388608.\n" 
"\t\td=224 -> (r=1152,c=448), d=256 -> (r=1088,c=512),\n"
"\t\td=384 -> (r=832, c=768), d=512 -> (r=576,c=1024).\n"
"\t\tOtherwise: (r=1024,c=576).\n"
"\t-Mnn\tTree hashing mode: 0 = LI, 1 = FNG\n"
"\t-Hnn\tMaximal height of the tree in LI mode. 0 <= H <= 6.\n"
"\t-Bnnnn\tBlock size per node in bytes,\n"
"\t\t1024, 2048, 4096, 8192, 16384, 32768. Default: 1024\n"
"\t-Dn\tNode degree, 1,2,4,8. Default: 8\n";

#define DIGEST_LENGTH  1024
#define HASH_MODE      0
#define MAX_HEIGHT     6
#define NODE_BLOCKSIZE 1024
#define NODE_DEGREE    8

int
main(int argc, const char** argv) {

    int i;    
    int digest_length  = DIGEST_LENGTH;
    int hash_mode      = HASH_MODE;
    int max_height     = MAX_HEIGHT;
    int node_blocksize = NODE_BLOCKSIZE;
    int node_degree    = NODE_DEGREE;

    bool opts = true;

    for(i=1;i<argc;++i) {
        if(argv[i][0] == (char)'-' && opts) {
            switch(argv[i][1]) {
            case (char)'h': {
                printf("%s\n", help);
                exit(EXIT_SUCCESS);
            } 
            case (char)'M': {
                int const hm = argv[i][2]-0x30;
                if(hm == 0  || hm == 1) { 
                    hash_mode = hm;
                } else {
                    fprintf(stderr,"Hash mode not supported: %d\n",hm);
                } break; 
            }
            case (char)'d': {
                errno = 0;
                int const d = (int)strtol(&argv[i][2], (char **)NULL, 10);
                if(errno != ERANGE && d > 0 && d < 8388609) {
                    digest_length = d;
                } else {
                    fprintf(stderr,"Digest length not supported:%d\n",d);
                } break; 
            }
            case (char)'B': {
                errno = 0;
                int const b = (int)strtol(&argv[i][2], (char **)NULL, 10);
                if(errno != ERANGE && (b==1024 || b==2048 || b==4096 ||
                                       b==8192 || b==16384|| b==32768)) {
                    node_blocksize = b;
                } else {
                    fprintf(stderr,"Node blocksize not supported:%d\n",b);
                } break;
            }
            case (char)'H': {
                errno = 0;
                int const h = (int)strtol(&argv[i][2], (char **)NULL, 10);
                if(errno != ERANGE && h>=0 && h<=6 ) {
                    max_height = h;
                } else {
                    fprintf(stderr,"Max. height of tree not supported:%d\n",h);
                } break;
            }
            case (char)'D': {
                errno = 0;
                int const d = (int)strtol(&argv[i][2], (char **)NULL, 10);
                if(errno != ERANGE && (d==1 || d==2 || d==4 || d==8)) {
                    node_degree = d;
                } else {
                    fprintf(stderr,"Node degree not supported:%d\n",d);
                } break;
            }}
        } else {
    
            printf("Hashtree mode ..... %s\n", hash_mode == 0 ? "LI" : "FNG"); 
            printf("Digest length ..... %d\n", digest_length); 
            printf("Node blocksize .... %d\n", node_blocksize); 
            printf("Max. tree height .. %d\n", max_height); 
            printf("Node degree ....... %d\n", node_degree); 
            
            opts = false;
            //TODO: mehrere files: char *fname[argc-i];
            char const *fname = argv[i];
            printf("%s\n", fname);
            
        }
    }

    //printf("%s\n",help_string);

    //return 0;
    CUDA_SAFE_CALL(cudaSetDevice(0));

    call_keccak_kernel();
    
    shrEXIT(argc, argv);
}

/********************************** end-of-file ******************************/

