#include "lbm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Default values for run control parameters */
int NSTEPS = 20001;
int OUTPUT_STEP = 2000;
bool GENERATE_IMAGES = true;

void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -n, --nsteps <number>     Number of simulation steps (default: %d)\n", NSTEPS);
    printf("  -o, --output-step <number> Output every N steps (default: %d)\n", OUTPUT_STEP);
    printf("  -i, --images <0|1>        Generate images: 1=yes, 0=no (default: %d)\n", GENERATE_IMAGES ? 1 : 0);
    printf("  -h, --help                Show this help message\n");
    printf("\nExample: %s -n 10000 -o 1000 -i 1\n", program_name);
}

int parse_arguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 1; // Help requested
        }
        else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--nsteps") == 0) {
            if (i + 1 < argc) {
                NSTEPS = atoi(argv[++i]);
                if (NSTEPS <= 0) {
                    fprintf(stderr, "Error: NSTEPS must be positive\n");
                    return -1;
                }
            } else {
                fprintf(stderr, "Error: -n/--nsteps requires a number\n");
                return -1;
            }
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output-step") == 0) {
            if (i + 1 < argc) {
                OUTPUT_STEP = atoi(argv[++i]);
                if (OUTPUT_STEP <= 0) {
                    fprintf(stderr, "Error: OUTPUT_STEP must be positive\n");
                    return -1;
                }
            } else {
                fprintf(stderr, "Error: -o/--output-step requires a number\n");
                return -1;
            }
        }
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--images") == 0) {
            if (i + 1 < argc) {
                int img_flag = atoi(argv[++i]);
                if (img_flag == 0) {
                    GENERATE_IMAGES = false;
                } else if (img_flag == 1) {
                    GENERATE_IMAGES = true;
                } else {
                    fprintf(stderr, "Error: -i/--images must be 0 or 1\n");
                    return -1;
                }
            } else {
                fprintf(stderr, "Error: -i/--images requires 0 or 1\n");
                return -1;
            }
        }
        else {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            fprintf(stderr, "Use -h or --help for usage information\n");
            return -1;
        }
    }
    return 0; // Success
}