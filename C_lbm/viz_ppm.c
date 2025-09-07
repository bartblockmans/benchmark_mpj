#include "lbm.h"
#include "viz_ppm.h"
#include <stdio.h>
#include <math.h>
#include <sys/stat.h>
#ifdef _WIN32
  #include <direct.h>
#endif

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* Linear blend in RdBu-like diverging scheme (blue->white->red) */
static void rdBu_color(float t, int *r, int *g, int *b) {
  float r1,g1,b1, r2,g2,b2, u;
  if (t <= 0.5f) {
    u = t/0.5f;
    r1=49.0f;  g1=54.0f;  b1=149.0f;   /* blue */
    r2=255.0f; g2=255.0f; b2=255.0f;   /* white */
  } else {
    u = (t-0.5f)/0.5f;
    r1=255.0f; g1=255.0f; b1=255.0f;   /* white */
    r2=165.0f; g2=0.0f;   b2=38.0f;    /* red */
  }
  int Rc = (int)lroundf((1.0f-u)*r1 + u*r2);
  int Gc = (int)lroundf((1.0f-u)*g1 + u*g2);
  int Bc = (int)lroundf((1.0f-u)*b1 + u*b2);
  *r = clampi(Rc,0,255); *g = clampi(Gc,0,255); *b = clampi(Bc,0,255);
}

static void ensure_images_dir(void) {
  struct stat st;
  if (stat("images", &st) != 0) {
#ifdef _WIN32
    _mkdir("images");
#else
    mkdir("images", 0755);
#endif
  }
}

static inline int wrap(int i, int n) { int r = i % n; return (r < 0) ? r + n : r; }

static void draw_circle_outline(int img[NY][NX][3], int xc, int yc, int radius) {
  const int N = 1024;
  for (int k = 0; k < N; ++k) {
    float th = 6.283185307179586f * (float)k / (float)N;
    int xi = (int)lroundf((float)xc + (float)radius * cosf(th));
    int yi = (int)lroundf((float)yc + (float)radius * sinf(th));
    if (xi >= 0 && xi < NX && yi >= 0 && yi < NY) {
      img[yi][xi][0] = 0; img[yi][xi][1] = 0; img[yi][xi][2] = 0;
    }
  }
}

void plot_vorticity_ppm(int step, int picnum) {
  (void)step;
  float vort[NY][NX];
  int   img [NY][NX][3];

  /* vorticity = d(uy)/dx - d(ux)/dy with periodic neighbors */
  for (int j = 0; j < NX; ++j) {
    int jm = wrap(j-1, NX), jp = wrap(j+1, NX);
    for (int i = 0; i < NY; ++i) {
      int im = wrap(i-1, NY), ip = wrap(i+1, NY);
      float right = UY[i][jp], left  = UY[i][jm];
      float up    = UX[ip][j], down  = UX[im][j];
      vort[i][j]  = (right - left) - (up - down);
    }
  }

  /* Fixed color limits as in your Python */
  const float zmin = -0.02f, zmax = 0.02f;
  const float invspan = 1.0f / (zmax - zmin);

  /* Map to colors */
  for (int j = 0; j < NX; ++j) {
    for (int i = 0; i < NY; ++i) {
      float t = (vort[i][j] - zmin) * invspan;
      if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
      int r,g,b; rdBu_color(t, &r,&g,&b);
      img[i][j][0] = r; img[i][j][1] = g; img[i][j][2] = b;
    }
  }

  /* Paint cylinder interior black */
  for (int j = 0; j < NX; ++j)
    for (int i = 0; i < NY; ++i)
      if (CYL[i][j]) { img[i][j][0]=0; img[i][j][1]=0; img[i][j][2]=0; }

  /* Outline */
  draw_circle_outline(img, CX0, CY0, RADIUS);

  /* Save PPM (ASCII P3) */
  ensure_images_dir();
  char fname[256];
  snprintf(fname, sizeof(fname), "images/lattice_boltzmann_%04d_c.ppm", picnum);
  FILE *f = fopen(fname, "w");
  if (!f) { perror("fopen PPM"); return; }
  fprintf(f, "P3\n%d %d\n255\n", NX, NY);
  /* write top->bottom so it matches matplotlib(origin='lower') look */
  for (int i = NY-1; i >= 0; --i) {
    for (int j = 0; j < NX; ++j) {
      fprintf(f, "%d %d %d\n", img[i][j][0], img[i][j][1], img[i][j][2]);
    }
  }
  fclose(f);
}