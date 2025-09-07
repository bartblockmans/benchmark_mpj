#ifndef NBODY_H
#define NBODY_H

#include <stdbool.h>
#include <stddef.h>

/* ================== Simulation parameters (match your Fortran) ================== */
#define N               4000          /* particles */
#define DT              1.0e-3f       /* time step */
#define GCONST          1.0f          /* gravitational constant */
#define SOFTENING       1.5e-2f       /* Plummer softening */
#define SEED_DEFAULT    17
#define SCENARIO_NAME   "galaxy_spiral"
/* Rendering */
#define IMG_SIZE        1024
#define BOUNDS          3.5f          /* world coords mapped to [-BOUNDS, +BOUNDS] */
#define OUTPUT_STEP_DEF 100

/* ================== API ================== */

/* JSON import (single-file naive parser; no deps) */
bool import_initial_conditions(
  const char* filename,
  float* x, float* y, float* z,
  float* u, float* v, float* w,
  float* m, int n);

/* Initial condition generator */
void init_ic_galaxy_spiral(int n, int seed,
  float* x, float* y, float* z,
  float* u, float* v, float* w,
  float* m);

/* Physics */
void compute_acc_sym_inplace(
  float* ax, float* ay, float* az,
  const float* x, const float* y, const float* z,
  const float* m, float G, float eps2, int n);

/* Energies */
float kinetic_energy(const float* m, const float* u, const float* v, const float* w, int n);
float potential_energy(const float* x, const float* y, const float* z, const float* m,
                       float G, float eps, int n);

/* Center-of-mass frame */
void center_of_mass_frame(float* x, float* y, float* z,
                          float* u, float* v, float* w,
                          const float* m, int n);

/* Visualization (PPM, ASCII P3) */
void save_snapshot(int step,
                   const float* x, const float* y, const float* z,
                   const float* u, const float* v, const float* w,
                   int n, const char* title);

/* Utilities */
void* xmalloc(size_t bytes);

/* RNG (deterministic per-seed helpers, similar spirit to your Fortran) */
float uniform_random(float a, float b, unsigned int seed);
float normal_random(unsigned int seed);                      /* Box–Muller */
float gamma_random(float shape, float scale, unsigned int seed);

#endif /* NBODY_H */
