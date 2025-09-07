#include "nbody.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char** argv) {
  /* Defaults (match your Fortran) */
  float tEnd = 1.0f;
  int   OUTPUT_STEP = OUTPUT_STEP_DEF;
  int   GENERATE_IMAGE = 1;
  char  IMPORT_IC[512] = "nbody_ic_galaxy_spiral_N4000.json";

  if (argc > 1 && (strcmp(argv[1], "-h")==0 || strcmp(argv[1], "--help")==0)) {
    printf("Usage: ./nbody [tEnd] [OUTPUT_STEP] [GENERATE_IMAGE] [IMPORT_IC]\n");
    printf("  tEnd: total simulation time (default 1.0)\n");
    printf("  OUTPUT_STEP: steps between images (default 100; 0 to disable)\n");
    printf("  GENERATE_IMAGE: 1=on, 0=off (default 1)\n");
    printf("  IMPORT_IC: JSON filename or '0' to generate (default: %s)\n", IMPORT_IC);
    return 0;
  }

  if (argc >= 2) tEnd = (float)atof(argv[1]);
  if (argc >= 3) OUTPUT_STEP = atoi(argv[2]);
  if (argc >= 4) GENERATE_IMAGE = atoi(argv[3]);
  if (argc >= 5) { strncpy(IMPORT_IC, argv[4], sizeof(IMPORT_IC)-1); IMPORT_IC[sizeof(IMPORT_IC)-1] = '\0'; }

  if (tEnd <= 0.0f) { fprintf(stderr, "Error: tEnd must be > 0\n"); return 1; }
  if (GENERATE_IMAGE && OUTPUT_STEP < 1) { fprintf(stderr, "Error: OUTPUT_STEP must be >= 1 when images on\n"); return 1; }

  printf("N-Body Galaxy Simulation in C (single-language)\n");
  printf("===============================================\n");
  printf("N=%d  dt=%g  G=%g  softening=%g  OUTPUT_STEP=%d  images=%s  IC=%s\n",
         N, (double)DT, (double)GCONST, (double)SOFTENING, OUTPUT_STEP, GENERATE_IMAGE?"on":"off", IMPORT_IC);

  /* Allocate arrays */
  float *x=(float*)xmalloc(N*sizeof(float)), *y=(float*)xmalloc(N*sizeof(float)), *z=(float*)xmalloc(N*sizeof(float));
  float *u=(float*)xmalloc(N*sizeof(float)), *v=(float*)xmalloc(N*sizeof(float)), *w=(float*)xmalloc(N*sizeof(float));
  float *m=(float*)xmalloc(N*sizeof(float));
  float *ax=(float*)xmalloc(N*sizeof(float)), *ay=(float*)xmalloc(N*sizeof(float)), *az=(float*)xmalloc(N*sizeof(float));

  /* IC: either generate or import JSON */
  if (strcmp(IMPORT_IC, "0") == 0) {
    printf("Generating initial conditions for scenario: %s\n", SCENARIO_NAME);
    init_ic_galaxy_spiral(N, SEED_DEFAULT, x,y,z, u,v,w, m);
  } else {
    printf("Importing initial conditions from: %s\n", IMPORT_IC);
    if (!import_initial_conditions(IMPORT_IC, x,y,z, u,v,w, m, N)) {
      fprintf(stderr, "Falling back to generated ICs.\n");
      init_ic_galaxy_spiral(N, SEED_DEFAULT, x,y,z, u,v,w, m);
    }
  }

  /* Initial forces */
  printf("Computing initial forces...\n");
  compute_acc_sym_inplace(ax,ay,az, x,y,z, m, GCONST, SOFTENING*SOFTENING, N);

  /* initial snapshot */
  if (GENERATE_IMAGE) {
    printf("Saving initial snapshot...\n");
    save_snapshot(0, x,y,z, u,v,w, N, "Initial");
  }

  /* main loop */
  const int Nt = (int)ceil(tEnd / DT);
  printf("Starting simulation for %d steps...\n\n", Nt);

  float t = 0.0f;
  for (int it=1; it<=Nt; ++it) {
    if (it % 100 == 0) printf("  Step %d / %d (t=%g)\n", it, Nt, (double)t);

    /* half-kick */
    for (int i=0;i<N;++i) { u[i] += 0.5f * DT * ax[i]; v[i] += 0.5f * DT * ay[i]; w[i] += 0.5f * DT * az[i]; }
    /* drift */
    for (int i=0;i<N;++i) { x[i] += DT * u[i]; y[i] += DT * v[i]; z[i] += DT * w[i]; }
    /* new accelerations */
    compute_acc_sym_inplace(ax,ay,az, x,y,z, m, GCONST, SOFTENING*SOFTENING, N);
    /* half-kick */
    for (int i=0;i<N;++i) { u[i] += 0.5f * DT * ax[i]; v[i] += 0.5f * DT * ay[i]; w[i] += 0.5f * DT * az[i]; }

    t += DT;

    if (GENERATE_IMAGE && OUTPUT_STEP>0 && (it % OUTPUT_STEP)==0) {
      save_snapshot(it, x,y,z, u,v,w, N, "Step");
    }
  }

  /* final snapshot */
  if (GENERATE_IMAGE) {
    printf("Saving final snapshot...\n");
    save_snapshot(Nt, x,y,z, u,v,w, N, "Final");
  }

  /* energies */
  float KE = kinetic_energy(m, u,v,w, N);
  float PE = potential_energy(x,y,z, m, GCONST, SOFTENING, N);
  float Etot = KE + PE;

  printf("\nSimulation completed!\n");
  printf("Final energies:\n");
  printf("  KE = %g\n  PE = %g\n  Etot = %g\n", (double)KE, (double)PE, (double)Etot);
  printf("Images in ./images (if enabled)\n");

  free(x); free(y); free(z); free(u); free(v); free(w); free(m); free(ax); free(ay); free(az);
  return 0;
}
