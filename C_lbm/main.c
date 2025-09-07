#include "lbm.h"
#include "viz_ppm.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
  /* Parse command line arguments */
  int parse_result = parse_arguments(argc, argv);
  if (parse_result == 1) {
    return 0; // Help was requested and printed
  } else if (parse_result == -1) {
    return 1; // Error in parsing
  }

  printf("LBM with Half-way Bounce-back (C, Python-matching)\n");
  printf("Relaxation time = %.8f\n", (double)TAU);
  printf("Domain size = %d x %d\n", NX, NY);
  printf("Umax = %.8f\n", (double)UMAX);
  printf("Simulation steps = %d\n", NSTEPS);
  printf("Output step = %d\n", OUTPUT_STEP);
  printf("Generate images = %s\n", GENERATE_IMAGES ? "Yes" : "No");

  /* Init */
  init_cylinder();
  initialize_flow();

  printf("Starting simulation for %d steps...\n", NSTEPS);
  int pic = 1;

  for (int step = 0; step < NSTEPS; ++step) {
    if (step % 1000 == 0) {
      float ux_min=1e9f, ux_max=-1e9f, uy_min=1e9f, uy_max=-1e9f, rho_min=1e9f, rho_max=-1e9f;
      for (int i=0;i<NY;++i) for (int j=0;j<NX;++j) {
        if (UX[i][j]<ux_min) ux_min=UX[i][j]; if (UX[i][j]>ux_max) ux_max=UX[i][j];
        if (UY[i][j]<uy_min) uy_min=UY[i][j]; if (UY[i][j]>uy_max) uy_max=UY[i][j];
        if (RHO[i][j]<rho_min) rho_min=RHO[i][j]; if (RHO[i][j]>rho_max) rho_max=RHO[i][j];
      }
      printf("Step %6d / %6d | UX [%g, %g]  UY [%g, %g]  RHO [%g, %g]\n",
             step, NSTEPS, ux_min, ux_max, uy_min, uy_max, rho_min, rho_max);
    }

    /* 1) periodic BC */
    apply_periodic_boundary_conditions();

    /* 2) streaming step: particles move along their velocity directions */
    streaming_step();

    /* 3) handle cylinder boundary (no-slip condition) */
    handle_cylinder_boundary();

    /* 4) macros */
    compute_macroscopic_variables();

    /* 5) collision step: relaxation toward equilibrium */
    collision_step();

    /* 6) inlet/outlet BC */
    apply_inflow_outflow_boundary_conditions();

    /* 7) viz */
    if (GENERATE_IMAGES && step % OUTPUT_STEP == 0) {
      plot_vorticity_ppm(step, pic);
      ++pic;
    }
  }

  printf("Simulation completed successfully!\n");
  if (GENERATE_IMAGES) {
    printf("Images saved in ./images\n");
  }
  return 0;
}