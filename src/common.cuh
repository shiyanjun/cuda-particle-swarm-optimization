#define DEBUG true

// Number of particles
int numberOfParticles = 1000;

// Window size
int width = 1024, height = 640;

// Particle size
int particleSize = 1;

// Cell size for uniform space partitioning
double cellSize = 5.0;

// Center, zoom
double centerX = 0.0;
double centerY = 0.0;
double zoomX = 10000.0;
double zoomY = zoomX * height / width;

// Time
double timeValue = 0.0;
double timeStep = 0.001;

// Automatic centering (press key Q to switch)
bool autoCenter = true;

// CUDA grid
dim3 blocks_2d(16, 16);
dim3 threads_2d(32, 32);
dim3 blocks_1d(16);
dim3 threads_1d(1024);

// CUDA resource for OpenGL output
struct cudaGraphicsResource *res;

#define CSC(call) {                                                        \
  cudaError err = call;                                                    \
  if (err != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, \
    __LINE__, cudaGetErrorString(err));                                    \
    exit(1);                                                               \
  }                                                                        \
}                                                                          \
while (0)

void reshapeFunc(int w, int h) {
  width = w;
  height = h;
  zoomY = zoomX * height / width;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);

  glewInit();

  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

  CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);
  glutInitWindowSize(width, height);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glutSwapBuffers();
}

void keyboardFunc(unsigned char key, int xmouse, int ymouse) {
  switch (key) {
    case 'w':
      centerY -= 0.1f * zoomY; break;
    case 'a':
      centerX -= 0.1f * zoomX; break;
    case 's':
      centerY += 0.1f * zoomY; break;
    case 'd':
      centerX += 0.1f * zoomX; break;
    case 'q':
      autoCenter = autoCenter ? false : true; break;
    case 45:
      particleSize -= 1; break;
    case 61:
      particleSize += 1; break;
    default: break;
  }

  if (particleSize < 0) particleSize = 0;
  if (particleSize > 100) particleSize = 100;
}

void mouseWheelFunc(int wheel, int direction, int x, int y) {
  zoomX += direction < 0 ? 0.1 * zoomX : -0.1 * zoomX;

  if (zoomX <= 0.01) zoomX = 0.01;
  if (zoomX > 1000000) zoomX = 1000000;
  zoomY = zoomX * height / width;
}
