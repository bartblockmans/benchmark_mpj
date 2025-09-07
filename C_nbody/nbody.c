#include "nbody.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
  #include <direct.h>
#endif

/* ============================== Utilities ============================== */

void* xmalloc(size_t bytes) {
  void* p = malloc(bytes);
  if (!p) { fprintf(stderr, "Out of memory (%zu bytes)\n", bytes); exit(1); }
  return p;
}

static inline int wrapi(int i, int n) { int r = i % n; return (r < 0) ? r + n : r; }

/* ============================== RNG ============================== */
/* Simple xorshift32 per-call seeded engine -> reproducible per seed */
static inline unsigned int xorshift32(unsigned int* s) {
  unsigned int x = *s;
  if (x == 0) x = 0x9E3779B9u;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}

static inline float urand01(unsigned int* s) {
  /* 24-bit mantissa to (0,1) */
  return (xorshift32(s) >> 8) * (1.0f / 16777216.0f);
}

float uniform_random(float a, float b, unsigned int seed) {
  unsigned int s = seed;
  float u = urand01(&s);
  return a + (b - a) * u;
}

float normal_random(unsigned int seed) {
  unsigned int s = seed ^ 0xA5A5A5A5u;
  float u1 = urand01(&s);
  float u2 = urand01(&s);
  if (u1 < 1e-12f) u1 = 1e-12f;
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

float gamma_random(float shape, float scale, unsigned int seed) {
  /* Approx like your Fortran: integer part = sum of exponentials; fractional part via -ln(u)*(frac) */
  int k = (int)floorf(shape);
  float frac = shape - (float)k;
  unsigned int s = seed ^ 0x51ED270Bu;
  float g = 0.0f;

  for (int i = 0; i < k; ++i) {
    float u = urand01(&s); if (u < 1e-12f) u = 1e-12f;
    g += -logf(u);
  }
  if (frac > 1e-6f) {
    float u = urand01(&s); if (u < 1e-12f) u = 1e-12f;
    g += -logf(u) * frac;
  }
  return g * scale;
}

/* ============================== JSON Import ============================== */

/* read whole file to string (NUL-terminated) */
static char* read_text_file(const char* filename, size_t* out_len) {
  FILE* f = fopen(filename, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long L = ftell(f);
  if (L < 0) { fclose(f); return NULL; }
  fseek(f, 0, SEEK_SET);
  char* buf = (char*)xmalloc((size_t)L + 1);
  size_t rd = fread(buf, 1, (size_t)L, f);
  fclose(f);
  buf[rd] = '\0';
  if (out_len) *out_len = rd;
  return buf;
}

static int parse_array_numbers(const char* s, const char* key, int n, float* out) {
  /* Find key, then first '[' and parse numbers until ']' or n items */
  const char* p = strstr(s, key);
  if (!p) return 0;
  const char* b = strchr(p, '[');
  if (!b) return 0;
  int count = 0;
  const char* cur = b + 1;
  while (*cur && *cur != ']') {
    /* skip to start of number, incl. minus/dot/exponent */
    while (*cur && !(*cur=='-' || *cur=='+' || *cur=='.' || (*cur>='0'&&*cur<='9'))) {
      if (*cur == ']') break;
      ++cur;
    }
    if (*cur == ']' || !*cur) break;
    char* endp = NULL;
    float val = strtof(cur, &endp);
    if (cur == endp) break; /* no progress */
    if (count < n) out[count++] = val;
    cur = endp;
    /* skip commas/space */
    while (*cur && *cur != ']' && *cur != '-' && *cur != '+' && *cur != '.' && !(*cur>='0'&&*cur<='9')) ++cur;
  }
  return count;
}

static int parse_int_field(const char* s, const char* key, int* out) {
  const char* p = strstr(s, key);
  if (!p) return 0;
  char* endp = NULL;
  long v = strtol(p, &endp, 10);
  if (p == endp) return 0;
  *out = (int)v;
  return 1;
}

static int parse_float_field(const char* s, const char* key, float* out) {
  const char* p = strstr(s, key);
  if (!p) return 0;
  char* endp = NULL;
  float v = strtof(p, &endp);
  if (p == endp) return 0;
  *out = v;
  return 1;
}

bool import_initial_conditions(
  const char* filename,
  float* x, float* y, float* z,
  float* u, float* v, float* w,
  float* m, int n)
{
  size_t L = 0;
  char* text = read_text_file(filename, &L);
  if (!text) {
    fprintf(stderr, "Error: could not open '%s'\n", filename);
    return false;
  }

  /* metadata (optional, for printing) */
  int N_actual = 0, seed_actual = SEED_DEFAULT;
  float G_actual = GCONST, soft_actual = SOFTENING;
  (void)parse_int_field(text, "\"N\"", &N_actual);
  (void)parse_int_field(text, "\"seed\"", &seed_actual);
  (void)parse_float_field(text, "\"Gconst\"", &G_actual);
  (void)parse_float_field(text, "\"softening\"", &soft_actual);

  /* arrays – allow either flat keys inside sections or direct top-level arrays */
  int cx = parse_array_numbers(text, "\"x\"", n, x);
  int cy = parse_array_numbers(text, "\"y\"", n, y);
  int cz = parse_array_numbers(text, "\"z\"", n, z);
  int cu = parse_array_numbers(text, "\"u\"", n, u);
  int cv = parse_array_numbers(text, "\"v\"", n, v);
  int cw = parse_array_numbers(text, "\"w\"", n, w);

  /* masses may be in "m" or "masses" */
  int cm = parse_array_numbers(text, "\"m\"", n, m);
  if (cm < n) cm = parse_array_numbers(text, "\"masses\"", n, m);

  free(text);

  if (cx<n || cy<n || cz<n || cu<n || cv<n || cw<n || cm<n) {
    fprintf(stderr, "Error: JSON arrays missing or too short (x:%d y:%d z:%d u:%d v:%d w:%d m:%d, need %d)\n",
            cx,cy,cz,cu,cv,cw,cm,n);
    return false;
  }

  printf("Imported initial conditions from: %s\n", filename);
  printf("  Particles: %d  G: %g  softening: %g  seed: %d\n",
         N_actual, (double)G_actual, (double)soft_actual, seed_actual);
  return true;
}

/* ============================== Initial conditions ============================== */

void center_of_mass_frame(float* x, float* y, float* z,
                          float* u, float* v, float* w,
                          const float* m, int n)
{
  (void)x; (void)y; (void)z;  /* positions untouched here, as in your Fortran */
  double mu=0.0, mv=0.0, mw=0.0, mbar=0.0;
  for (int i=0;i<n;++i) { mu += m[i]*u[i]; mv += m[i]*v[i]; mw += m[i]*w[i]; mbar += m[i]; }
  if (mbar == 0.0) return;
  float ucm = (float)(mu/mbar), vcm=(float)(mv/mbar), wcm=(float)(mw/mbar);
  for (int i=0;i<n;++i) { u[i]-=ucm; v[i]-=vcm; w[i]-=wcm; }
}

static void generate_spiral_disk(int n, float mass_total,
  float* x, float* y, float* z,
  float* u, float* v, float* w, float* m,
  int seed, float phi0)
{
  const float Rd = 0.55f, Rmax = 1.7f;
  const float m_val = 2.0f;
  const float pitch_deg = 18.0f;
  const float arm_amp = 0.70f;   (void)arm_amp; /* not used explicitly */
  const float z_thick = 0.07f;
  const float v0 = 1.05f, v_rise = 0.32f;
  const float nudge_r = 0.06f, nudge_t = 0.03f, jitter = 0.025f;

  const float k_spiral = 1.0f / tanf(pitch_deg * 3.14159265358979323846f / 180.0f);

  for (int i=0;i<n;++i) {
    float R = gamma_random(2.0f, Rd, (unsigned int)(seed + i));
    if (R > Rmax) R = gamma_random(2.0f, Rd, (unsigned int)(seed + i + 1000));
    float theta = uniform_random(0.0f, 2.0f*3.14159265358979323846f, (unsigned int)(seed + i + 2000));

    x[i] = R * cosf(theta);
    y[i] = R * sinf(theta);
    z[i] = z_thick * 0.5f * normal_random((unsigned int)(seed + i + 3000));

    float v_circ = v0 * tanhf(R / v_rise);
    float phase = m_val * (theta - k_spiral * logf(R + 1e-6f) - phi0);

    float v_r = nudge_r * v_circ * cosf(phase);
    float v_t = v_circ * (1.0f + nudge_t * sinf(phase));

    u[i] = -v_t * sinf(theta) + v_r * cosf(theta);
    v[i] =  v_t * cosf(theta) + v_r * sinf(theta);
    w[i] = 0.5f * jitter * normal_random((unsigned int)(seed + i + 4000));

    u[i] += jitter * normal_random((unsigned int)(seed + i + 5000));
    v[i] += jitter * normal_random((unsigned int)(seed + i + 6000));

    m[i] = mass_total / (float)n;
  }
}

void init_ic_galaxy_spiral(int n, int seed,
  float* x, float* y, float* z,
  float* u, float* v, float* w,
  float* m)
{
  int N1 = n/2, N2 = n - N1;
  float *x1=(float*)xmalloc(N1*sizeof(float)), *y1=(float*)xmalloc(N1*sizeof(float)), *z1=(float*)xmalloc(N1*sizeof(float));
  float *u1=(float*)xmalloc(N1*sizeof(float)), *v1=(float*)xmalloc(N1*sizeof(float)), *w1=(float*)xmalloc(N1*sizeof(float)), *m1=(float*)xmalloc(N1*sizeof(float));
  float *x2=(float*)xmalloc(N2*sizeof(float)), *y2=(float*)xmalloc(N2*sizeof(float)), *z2=(float*)xmalloc(N2*sizeof(float));
  float *u2=(float*)xmalloc(N2*sizeof(float)), *v2=(float*)xmalloc(N2*sizeof(float)), *w2=(float*)xmalloc(N2*sizeof(float)), *m2=(float*)xmalloc(N2*sizeof(float));

  generate_spiral_disk(N1, 10.0f, x1,y1,z1, u1,v1,w1, m1, seed, 0.0f);
  float phi2 = 3.14159f / 3.0f;
  generate_spiral_disk(N2, 10.0f, x2,y2,z2, u2,v2,w2, m2, seed, phi2);

  /* reverse vel of galaxy 2 and offset positions/velocities */
  for (int i=0;i<N2;++i) { u2[i] = -u2[i]; v2[i] = -v2[i]; }
  const float d = 2.1f, vcm = 0.45f;
  for (int i=0;i<N1;++i) { x1[i] -= d; u1[i] += vcm; }
  for (int i=0;i<N2;++i) { x2[i] += d; u2[i] -= vcm; }

  /* combine */
  memcpy(x, x1, N1*sizeof(float));  memcpy(y, y1, N1*sizeof(float));  memcpy(z, z1, N1*sizeof(float));
  memcpy(u, u1, N1*sizeof(float));  memcpy(v, v1, N1*sizeof(float));  memcpy(w, w1, N1*sizeof(float));  memcpy(m, m1, N1*sizeof(float));
  memcpy(x+N1, x2, N2*sizeof(float)); memcpy(y+N1, y2, N2*sizeof(float)); memcpy(z+N1, z2, N2*sizeof(float));
  memcpy(u+N1, u2, N2*sizeof(float)); memcpy(v+N1, v2, N2*sizeof(float)); memcpy(w+N1, w2, N2*sizeof(float)); memcpy(m+N1, m2, N2*sizeof(float));

  center_of_mass_frame(x,y,z,u,v,w,m,n);

  free(x1); free(y1); free(z1); free(u1); free(v1); free(w1); free(m1);
  free(x2); free(y2); free(z2); free(u2); free(v2); free(w2); free(m2);
}

/* ============================== Physics ============================== */

void compute_acc_sym_inplace(
  float* ax, float* ay, float* az,
  const float* x, const float* y, const float* z,
  const float* m, float G, float eps2, int n)
{
  for (int i=0;i<n;++i) { ax[i]=0.0f; ay[i]=0.0f; az[i]=0.0f; }

  for (int i=0;i<n-1;++i) {
    float xi=x[i], yi=y[i], zi=z[i];
    for (int j=i+1;j<n;++j) {
      float dx = x[j]-xi, dy = y[j]-yi, dz = z[j]-zi;
      float r2 = dx*dx + dy*dy + dz*dz + eps2;
      float inv_r = 1.0f / sqrtf(r2);
      float inv_r3 = inv_r / r2;
      float s = G * inv_r3;
      float fx = s * dx, fy = s * dy, fz = s * dz;
      ax[i] += m[j] * fx; ay[i] += m[j] * fy; az[i] += m[j] * fz;
      ax[j] -= m[i] * fx; ay[j] -= m[i] * fy; az[j] -= m[i] * fz;
    }
  }
}

float kinetic_energy(const float* m, const float* u, const float* v, const float* w, int n) {
  double ke = 0.0;
  for (int i=0;i<n;++i) ke += 0.5 * (double)m[i] * ( (double)u[i]*u[i] + (double)v[i]*v[i] + (double)w[i]*w[i] );
  return (float)ke;
}

float potential_energy(const float* x, const float* y, const float* z, const float* m,
                       float G, float eps, int n)
{
  double pe = 0.0;
  float eps2 = eps*eps;
  for (int i=0;i<n-1;++i) {
    for (int j=i+1;j<n;++j) {
      float dx = x[j]-x[i], dy = y[j]-y[i], dz = z[j]-z[i];
      float r = sqrtf(dx*dx + dy*dy + dz*dz + eps2);
      pe -= (double)G * (double)m[i] * (double)m[j] / (double)r;
    }
  }
  return (float)pe;
}

/* ============================== Visualization (PPM) ============================== */

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

/* Viridis (piecewise linear like your Fortran) */
static void viridis_color(float t, int* R, int* G, int* B) {
  float r1,g1,b1,r2,g2,b2,u;
  if (t <= 0.25f) {
    u=t/0.25f; r1=68; g1=1; b1=84;   r2=59; g2=82; b2=139;
  } else if (t <= 0.5f) {
    u=(t-0.25f)/0.25f; r1=59; g1=82; b1=139; r2=33; g2=144; b2=140;
  } else if (t <= 0.75f) {
    u=(t-0.5f)/0.25f; r1=33; g1=144; b1=140; r2=92; g2=200; b2=99;
  } else {
    u=(t-0.75f)/0.25f; r1=92; g1=200; b1=99; r2=253; g2=231; b2=37;
  }
  int Rc=(int)lroundf((1.0f-u)*r1 + u*r2);
  int Gc=(int)lroundf((1.0f-u)*g1 + u*g2);
  int Bc=(int)lroundf((1.0f-u)*b1 + u*b2);
  if (Rc<0) Rc=0; if (Rc>255) Rc=255;
  if (Gc<0) Gc=0; if (Gc>255) Gc=255;
  if (Bc<0) Bc=0; if (Bc>255) Bc=255;
  *R=Rc; *G=Gc; *B=Bc;
}

void save_snapshot(int step,
                   const float* x, const float* y, const float* z,
                   const float* u, const float* v, const float* w,
                   int n, const char* title)
{
  (void)title; (void)z; (void)w; /* 2D render in x–y plane, colored by speed */
  ensure_images_dir();

  /* Determine speed range for color mapping */
  float smin = 1e30f, smax = -1e30f;
  for (int i=0;i<n;++i) {
    float speed = sqrtf(u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
    if (speed < smin) smin = speed;
    if (speed > smax) smax = speed;
  }
  float span = (smax - smin);
  if (span < 1e-20f) span = 1.0f;

  /* Image buffer */
  int (*img)[IMG_SIZE][3] = xmalloc((size_t)IMG_SIZE * IMG_SIZE * 3 * sizeof(int));
  for (int i=0;i<IMG_SIZE;i++) for (int j=0;j<IMG_SIZE;j++) { img[i][j][0]=0; img[i][j][1]=0; img[i][j][2]=0; }

  int particles_plotted = 0;

  for (int p=0;p<n;++p) {
    float speed = sqrtf(u[p]*u[p] + v[p]*v[p] + w[p]*w[p]);
    float t = (speed - smin) / span;
    if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
    int R,G,B; viridis_color(t, &R,&G,&B);

    float fx = (x[p] + BOUNDS) / (2.0f*BOUNDS) * (IMG_SIZE - 1);
    float fy = (y[p] + BOUNDS) / (2.0f*BOUNDS) * (IMG_SIZE - 1);

    int xi = (int)lroundf(fx), yi = (int)lroundf(fy);
    /* draw 5x5 disc (radius^2 <= 4) */
    for (int dy=-2; dy<=2; ++dy) {
      for (int dx=-2; dx<=2; ++dx) {
        if (dx*dx + dy*dy <= 4) {
          int X = xi + dx, Y = yi + dy;
          if (X>=0 && X<IMG_SIZE && Y>=0 && Y<IMG_SIZE) {
            img[Y][X][0] = R; img[Y][X][1] = G; img[Y][X][2] = B;
          }
        }
      }
    }
    if (xi>=0 && xi<IMG_SIZE && yi>=0 && yi<IMG_SIZE) ++particles_plotted;
  }

  if (step == 0) {
    float xmin=1e30f,xmax=-1e30f,ymin=1e30f,ymax=-1e30f;
    for (int i=0;i<n;++i) { if (x[i]<xmin)xmin=x[i]; if (x[i]>xmax)xmax=x[i]; if (y[i]<ymin)ymin=y[i]; if (y[i]>ymax)ymax=y[i]; }
    printf("  Particles: %d, x=[%g,%g], y=[%g,%g], plotted=%d\n", n, xmin,xmax,ymin,ymax, particles_plotted);
  }

  char fname[256];
  snprintf(fname, sizeof(fname), "images/nbody_%04d_c.ppm", step);
  FILE* f = fopen(fname, "w");
  if (!f) { perror("fopen"); free(img); return; }
  fprintf(f, "P3\n%d %d\n255\n", IMG_SIZE, IMG_SIZE);
  for (int i=IMG_SIZE-1; i>=0; --i) {
    for (int j=0; j<IMG_SIZE; ++j) {
      fprintf(f, "%d %d %d\n", img[i][j][0], img[i][j][1], img[i][j][2]);
    }
  }
  fclose(f);
  free(img);
}
