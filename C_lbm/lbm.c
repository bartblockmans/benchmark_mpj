#include "lbm.h"
#include <math.h>
#include <string.h>   /* memcpy */
#include <stdio.h>

/* Globals */
float F[NY][NX][Q], F2[NY][NX][Q];
float RHO[NY][NX], UX[NY][NX], UY[NY][NX];
bool  CYL[NY][NX];
/* INCOMING_MASKS array removed - using original bounce-back approach */
float TMP2D[NY][NX];

/* D2Q9 constants (0-based, match your Python) */
const int   CX[Q]  = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
const int   CY[Q]  = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
const int   OPP[Q] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
const float W[Q]   = { 4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                       1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f };

static inline int wrap(int i, int n) { int r = i % n; return (r < 0) ? r + n : r; }

/* --- helpers --- */

/* roll_logical_dxdy function removed - no longer needed with original bounce-back approach */

/* --- API impl --- */

void init_cylinder(void) {
  for (int j = 0; j < NX; ++j) {
    for (int i = 0; i < NY; ++i) {
      int dx = j - CX0;
      int dy = i - CY0;
      CYL[i][j] = (dx*dx + dy*dy) <= RADIUS*RADIUS;
    }
  }
}

/* precompute_incoming_masks function removed - using original bounce-back approach */

void initialize_flow(void) {
  /* rho=1, ux=0 (except edges), uy=0; F=Feq(rho,u) */
  for (int i = 0; i < NY; ++i) {
    for (int j = 0; j < NX; ++j) {
      RHO[i][j] = 1.0f;
      UX[i][j]  = 0.0f;
      UY[i][j]  = 0.0f;
    }
    UX[i][0]     = UMAX;
    UX[i][NX-1]  = UMAX;
  }

  for (int k = 0; k < Q; ++k) {
    for (int j = 0; j < NX; ++j) {
      for (int i = 0; i < NY; ++i) {
        float cu  = CX[k]*UX[i][j] + CY[k]*UY[i][j];
        float usq = UX[i][j]*UX[i][j] + UY[i][j]*UY[i][j];
        float feq = RHO[i][j] * W[k] * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usq);
        F[i][j][k] = feq;
      }
    }
  }
}

void apply_periodic_boundary_conditions(void) {
  for (int i = 0; i < NY; ++i) {
    F[i][0][1]    = F[i][NX-1][1];   /* east at left edge (k=1)  */
    F[i][0][5]    = F[i][NX-1][5];   /* ne  at left edge (k=5)   */
    F[i][0][8]    = F[i][NX-1][8];   /* se  at left edge (k=8)   */
    F[i][NX-1][3] = F[i][0][3];      /* west at right edge (k=3) */
    F[i][NX-1][6] = F[i][0][6];      /* nw  at right edge (k=6)  */
    F[i][NX-1][7] = F[i][0][7];      /* sw  at right edge (k=7)  */
  }
  for (int j = 0; j < NX; ++j) {
    F[0][j][2]    = F[NY-1][j][2];   /* north at bottom edge (k=2) */
    F[0][j][5]    = F[NY-1][j][5];   /* ne    at bottom edge        */
    F[0][j][6]    = F[NY-1][j][6];   /* nw    at bottom edge        */
    F[NY-1][j][4] = F[0][j][4];      /* south at top edge (k=4)     */
    F[NY-1][j][7] = F[0][j][7];      /* sw    at top edge           */
    F[NY-1][j][8] = F[0][j][8];      /* se    at top edge           */
  }
}

void streaming_step(void) {
  /* Original streaming step: particles move along their velocity directions */
  for (int k = 0; k < Q; ++k) {
    int dx = CX[k], dy = CY[k];

    /* x shift into TMP2D */
    if (dx == 0) {
      for (int i = 0; i < NY; ++i)
        for (int j = 0; j < NX; ++j)
          TMP2D[i][j] = F[i][j][k];
    } else if (dx == 1) {
      for (int i = 0; i < NY; ++i) {
        for (int j = 1; j < NX; ++j) TMP2D[i][j] = F[i][j-1][k];
        TMP2D[i][0] = F[i][NX-1][k];
      }
    } else if (dx == -1) {
      for (int i = 0; i < NY; ++i) {
        for (int j = 0; j < NX-1; ++j) TMP2D[i][j] = F[i][j+1][k];
        TMP2D[i][NX-1] = F[i][0][k];
      }
    }

    /* y shift into F */
    if (dy == 0) {
      for (int i = 0; i < NY; ++i)
        for (int j = 0; j < NX; ++j)
          F[i][j][k] = TMP2D[i][j];
    } else if (dy == 1) {
      for (int i = 1; i < NY; ++i)
        for (int j = 0; j < NX; ++j)
          F[i][j][k] = TMP2D[i-1][j];
      for (int j = 0; j < NX; ++j) F[0][j][k] = TMP2D[NY-1][j];
    } else if (dy == -1) {
      for (int i = 0; i < NY-1; ++i)
        for (int j = 0; j < NX; ++j)
          F[i][j][k] = TMP2D[i+1][j];
      for (int j = 0; j < NX; ++j) F[NY-1][j][k] = TMP2D[0][j];
    }
  }
}

void handle_cylinder_boundary(void) {
  /* Original bounce-back method where particles hitting the cylinder reverse their direction */
  memcpy(F2, F, sizeof(F));  /* copy distribution functions */
  
  for (int k = 1; k < Q; ++k) {  /* skip rest particle (k=0) */
    for (int i = 0; i < NY; ++i) {
      for (int j = 0; j < NX; ++j) {
        if (CYL[i][j]) {
          /* Check if neighbor in direction -CX[k], -CY[k] is not solid */
          int ni = wrap(i - CY[k], NY);
          int nj = wrap(j - CX[k], NX);
          if (!CYL[ni][nj]) {
            /* Bounce back: reverse particle direction */
            F2[i][j][k] = F[i][j][OPP[k]];
          }
        }
      }
    }
  }
  
  memcpy(F, F2, sizeof(F));  /* update F with bounced-back values */
}

void compute_macroscopic_variables(void) {
  for (int i = 0; i < NY; ++i) {
    for (int j = 0; j < NX; ++j) {
      float rho = 0.0f, ux = 0.0f, uy = 0.0f;
      for (int k = 0; k < Q; ++k) rho += F[i][j][k];
      for (int k = 0; k < Q; ++k) { ux += F[i][j][k] * CX[k];  uy += F[i][j][k] * CY[k]; }
      UX[i][j] = ux / rho;
      UY[i][j] = uy / rho;
      RHO[i][j] = rho;
      if (CYL[i][j]) { UX[i][j] = 0.0f; UY[i][j] = 0.0f; }
    }
  }
}

void collision_step(void) {
  /* Original collision step using the BGK approximation */
  float Feq[NY][NX][Q];
  
  /* Compute equilibrium distribution functions */
  for (int k = 0; k < Q; ++k) {
    for (int i = 0; i < NY; ++i) {
      for (int j = 0; j < NX; ++j) {
        float cu = CX[k]*UX[i][j] + CY[k]*UY[i][j];
        float usq = UX[i][j]*UX[i][j] + UY[i][j]*UY[i][j];
        Feq[i][j][k] = RHO[i][j] * W[k] * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usq);
      }
    }
  }
  
  /* BGK collision: relax toward equilibrium */
  for (int k = 0; k < Q; ++k) {
    for (int i = 0; i < NY; ++i) {
      for (int j = 0; j < NX; ++j) {
        F[i][j][k] = F[i][j][k] - (1.0f/TAU) * (F[i][j][k] - Feq[i][j][k]);
      }
    }
  }
}

void apply_inflow_outflow_boundary_conditions(void) {
  /* Set boundary velocities */
  for (int i = 0; i < NY; ++i) {
    UX[i][0]     = UMAX;  UY[i][0]     = 0.0f;
    UX[i][NX-1]  = UMAX;  UY[i][NX-1]  = 0.0f;
  }

  /* Left edge (j=0) */
  for (int i = 0; i < NY; ++i) TMP2D[i][0] = UX[i][0]*UX[i][0]; /* usqL in TMP2D[:,0] */
  for (int k = 0; k < Q; ++k) {
    for (int i = 0; i < NY; ++i) {
      float cu = CX[k]*UX[i][0];
      float pol = 1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*TMP2D[i][0];
      F[i][0][k] = RHO[i][0] * W[k] * pol;
    }
  }

  /* Right edge (j=NX-1) */
  for (int i = 0; i < NY; ++i) TMP2D[i][NX-1] = UX[i][NX-1]*UX[i][NX-1]; /* usqR in TMP2D[:,NX-1] */
  for (int k = 0; k < Q; ++k) {
    for (int i = 0; i < NY; ++i) {
      float cu = CX[k]*UX[i][NX-1];
      float pol = 1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*TMP2D[i][NX-1];
      F[i][NX-1][k] = RHO[i][NX-1] * W[k] * pol;
    }
  }
}