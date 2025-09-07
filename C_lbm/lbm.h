#ifndef LBM_H
#define LBM_H

#include <stdbool.h>

#define NX 400
#define NY 100
#define Q  9

/* Geometry & flow */
#define CX0 70
#define CY0 50
#define RADIUS 20
#define UMAX 0.1f
#define TAU  0.56f

/* Run control - now configurable via command line */
extern int NSTEPS;
extern int OUTPUT_STEP;
extern bool GENERATE_IMAGES;

/* Globals (contiguous arrays for speed, single precision) */
extern float F[NY][NX][Q], F2[NY][NX][Q];
extern float RHO[NY][NX], UX[NY][NX], UY[NY][NX];
extern bool  CYL[NY][NX];
/* INCOMING_MASKS array removed - using original bounce-back approach */
extern float TMP2D[NY][NX];

/* Lattice constants (D2Q9) */
extern const int   CX[Q], CY[Q], OPP[Q];
extern const float W[Q];

/* API */
void init_cylinder(void);
void initialize_flow(void);
void apply_periodic_boundary_conditions(void);
void streaming_step(void);
void handle_cylinder_boundary(void);
void compute_macroscopic_variables(void);
void collision_step(void);
void apply_inflow_outflow_boundary_conditions(void);

/* Viz (PPM writer) */
void plot_vorticity_ppm(int step, int picnum);

/* Configuration */
void print_usage(const char* program_name);
int parse_arguments(int argc, char* argv[]);

#endif /* LBM_H */