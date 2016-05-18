/*
Particle swarm optimization
by Ivan Vinogradov
2016
*/

#include <iostream>
#include <chrono>
#include <cfloat>
#include <cmath>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// cuRAND
#include <curand.h>
#include <curand_kernel.h>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "particle.cuh"

#define CSC(call) {                                                        \
  cudaError err = call;                                                    \
  if (err != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, \
    __LINE__, cudaGetErrorString(err));                                    \
    exit(1);                                                               \
  }                                                                        \
}                                                                          \
while (0)

#define DEBUG false

#define GLOBAL_COEFF    0.000050   // coefficient of global solution
#define LOCAL_COEFF     0.00000010 // coefficient of local solution
#define RANDOM_COEFF    0.010      // coefficient of random motion
#define DAMPING_COEFF   0.99250    // coefficient of damping force
#define REPULSION_COEFF 0.10       // coefficient of repulsive force

// CUDA grid
dim3 blocks_2d(16, 16);
dim3 threads_2d(32, 32);
dim3 blocks_1d(16);
dim3 threads_1d(1024);

// Number of particles
int numberOfParticles = 1000;

// Window size
int width = 1024, height = 640;

// Particle size
int particleSize = 1;

// Cell size for uniform space partitioning
float cellSize = 5.0;

// Center, zoom
float centerX = 0.0;
float centerY = 0.0;
float zoomX = 10000.0;
float zoomY = zoomX * height / width;

// Time
float timeValue = 0.0;
float timeStep = 0.001;

// Automatic centering (press key Q to switch)
bool autoCenter = true;

// OpenGL buffer
GLuint vbo;

// CUDA resource for OpenGL output
struct cudaGraphicsResource *res;

Particle *devParticleArray;
ParticleArea *devPartileAreaArray;
curandState *devRandomState;

// Window size, center, zoom, function min-max, time
__constant__ int devWidth;
__constant__ int devHeight;
__constant__ int devParticleSize;
__constant__ float devCenterX;
__constant__ float devCenterY;
__constant__ float devZoomX;
__constant__ float devZoomY;
__constant__ float devTimeValue;
__constant__ float devTimeStep;

// Parameters of uniform space partitioning
__constant__ float devUniformSpaceMinX;
__constant__ float devUniformSpaceMaxX;
__constant__ float devUniformSpaceMinY;
__constant__ float devUniformSpaceMaxY;
__constant__ float devUniformSpaceCellSize;

// Screen coordinates into real coordinates
__device__
float2 indexToCoord(int2 index) {
  return make_float2(
    (2.0f * index.x / (float)(devWidth - 1) - 1.0f) * devZoomX + devCenterX,
    -(2.0f * index.y / (float)(devHeight - 1) - 1.0f) * devZoomY + devCenterY);
}

// Real coordinates into screen coordinates
__device__
int2 coordToIndex(float2 coord) {
  return make_int2(
    0.5f * (devWidth - 1) * (1.0f + (coord.x - devCenterX) / devZoomX),
    0.5f * (devHeight - 1) * (1.0f - (coord.y - devCenterY) / devZoomY)
  );
}

/*  Schwefel Function */
__device__
float fun(float2 coord) {
  return -coord.x * sin(sqrt(fabs(coord.x))) - coord.y * sin(sqrt(fabs(coord.y)));
}

__device__
float fun(int2 index) {
  return fun(indexToCoord(index));
}

__global__
void initRandomState(curandState *state, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetx = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += offsetx) {
    curand_init(1337, i, 0, &state[i]);
  }
}

__global__
void kernelSwarmInit(Particle *particles, int n, curandState *state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetx = blockDim.x * gridDim.x;
  Particle p;
  for (int i = idx; i < n; i += offsetx) {
    p = particles[i];

    // Position in the center of the screen
    p.coords = p.best_coords = indexToCoord(make_int2(devWidth / 2, devHeight / 2));

    // Random starting angle and the speed
    float angle = 2.0 * 3.14 * curand_uniform(&state[i]);
    float speed = 100.0 * curand_uniform(&state[i]);
    p.speed = make_float2(cos(angle) * speed, sin(angle) * speed);
    p.value = p.best_value = FLT_MAX;

    particles[i] = p;
  }
}

__global__
void kernelSwarmUpdate(uchar4 *image, Particle *particles, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetx = blockDim.x * gridDim.x;

  Particle p;

  for (int i = idx; i < n; i += offsetx) {
    p = particles[i];
    p.value = fun(p.coords);
    if (p.value < p.best_value) {
      p.best_value = p.value;
      p.best_coords = p.coords;
    }
    particles[i] = p;
  }
}

__global__
void kernelNormalizedHeatMap(uchar4 *heatMap, float minValue, float maxValue) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int offsetx = blockDim.x * gridDim.x;
  int offsety = blockDim.y * gridDim.y;
  int i, j;
  float f;
  for (i = idx; i < devWidth; i += offsetx) {
    for (j = idy; j < devHeight; j += offsety) {
      f = (fun(make_int2(i, j)) - minValue) / (maxValue - minValue);
      if (f < 0.0) f = 0.0; else if (f > 1.0) f = 1.0;
      heatMap[j * devWidth + i] = make_uchar4(
        (int)(f * 255), 0, (int)((1.0 - f) * 255), 255
      );
    }
  }
}

__global__
void kernelSwarmDraw(uchar4 *heatMap, Particle *particles, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetx = blockDim.x * gridDim.x;
  int size = devParticleSize;
  for (int i = idx; i < n; i += offsetx) {
    int2 index = coordToIndex(particles[i].coords);
    for (int x = index.x - size; x <= index.x + size; x++) {
      for (int y = index.y - size; y <= index.y + size; y++) {
        if (x >= 0 && x < devWidth && y >= 0 && y < devHeight &&
            (x - index.x) * (x - index.x) + (y - index.y) * (y - index.y) <= size * size
        ) {

          if (DEBUG && i == n / 2) {
            heatMap[y * devWidth + x] = make_uchar4(0, 255, 0, 255);
            continue;
          }

          heatMap[y * devWidth + x] = make_uchar4(255, 255, 255, 255);
        }
      }
    }
  }
}

// The assignment for each particle corresponding space partitioning cell
__global__
void kernelSwarmAssociateWithCells(Particle *particles, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetx = blockDim.x * gridDim.x;

  int sizeX = ceil((devUniformSpaceMaxX - devUniformSpaceMinX) / devUniformSpaceCellSize);
  int sizeY = ceil((devUniformSpaceMaxY - devUniformSpaceMinY) / devUniformSpaceCellSize);

  Particle p;
  int cellX, cellY;

  for (int i = idx; i < n; i += offsetx) {
    p = particles[i];
    cellX = (p.coords.x - devUniformSpaceMinX) / devUniformSpaceCellSize;
    cellY = (p.coords.y - devUniformSpaceMinY) / devUniformSpaceCellSize;
    p.cellIndex = cellX * sizeX + cellY;
    particles[i] = p;
  }
}

// The total force of repulsion for the i-th particle (without the space partitioning)
__device__
float2 calculateRepulsionAll(Particle *particles, int n, int i) {
  float2 repulsion, diff, coords_a, coords_b;
  float distance;

  // TODO
  float minDistance = FLT_MAX;

  repulsion.x = 0.0;
  repulsion.y = 0.0;
  coords_a = particles[i].coords;

  for (int j = 0; j < n; j++) {
    if (j == i) continue;
    coords_b = particles[j].coords;
    diff.x = coords_a.x - coords_b.x;
    diff.y = coords_a.y - coords_b.y;
    distance = sqrt(diff.x * diff.x + diff.y * diff.y);

    // TODO
    if (DEBUG && i == n / 2 && distance < minDistance) {
      minDistance = distance;
    }

    distance = pow(distance, 5);

    if (distance < 0.5) distance = 0.5;

    repulsion.x += diff.x / distance;
    repulsion.y += diff.y / distance;
  }

  // TODO
  if (DEBUG && i == n / 2) {
    printf("distance: %lf; interactions: %d\n", minDistance, n);
  }
  
  return repulsion;
}

// Binary search in a sorted array of particles by cell index of space partitioning
__device__
int binarySearchLowerBound(Particle *particles, int size, int cellIndex) {
    int left = 0, right = size - 1, middle;
    while (left <= right) {
        middle = left + (right - left) / 2;
        if ((&particles[middle])->cellIndex < cellIndex)
            left = middle + 1;
        else
            right = middle - 1;
    }
    return left;
}

// Check that the particle was found in binarySearchLowerBound
__device__
int isFound(Particle *particles, int size, int cellIndex, int index) {
    return index < size && particles[index].cellIndex == cellIndex;
}

// Search the first particle with specified cell partition index
__device__
int findParticleByCell(Particle *particles, int size, int cellIndex) {
  int index = binarySearchLowerBound(particles, size, cellIndex);
  return isFound(particles, size, cellIndex, index) ? index : -1;
}

// The total force of repulsion for the i-th particle (with the space partitioning)
__device__
float2 calculateRepulsionClosest(Particle *particles, int n, int i) {
  float2 diff, repulsion = make_float2(0.0, 0.0);
  float distance;

  // TODO
  float minDistance = FLT_MAX;

  // Counter of interacting particles
  int counter = 0;

  // Dimensions of the space partitioning
  int sizeX = ceil(abs(devUniformSpaceMaxX - devUniformSpaceMinX) / devUniformSpaceCellSize);
  int sizeY = ceil(abs(devUniformSpaceMaxY - devUniformSpaceMinY) / devUniformSpaceCellSize);

  if (sizeX < 1) sizeX = 1;
  if (sizeY < 1) sizeY = 1;

  Particle pa, pb;

  pa = particles[i];

  // TODO: деление на ноль
  int cellX = pa.cellIndex / sizeX;
  int cellY = pa.cellIndex % sizeX;

  int radius = 1;
  for (int x = cellX - radius; x <= cellX + radius; x++) {
    for (int y = cellY - radius; y <= cellY + radius; y++) {
      int neighborCellIndex = x * sizeX + y;
      int neighborIndex = findParticleByCell(particles, n, neighborCellIndex);
      if (neighborIndex != -1)
      {
        for (int k = neighborIndex; k < n; k++)
        {
          if (k == i) continue;
          pb = particles[k];
          if (pb.cellIndex != neighborCellIndex) break;

          diff.x = pa.coords.x - pb.coords.x;
          diff.y = pa.coords.y - pb.coords.y;

          distance = sqrt(diff.x * diff.x + diff.y * diff.y);

          // TODO
          if (DEBUG && i == n / 2 && distance < minDistance) {
            minDistance = distance;
          }

          distance = pow(distance, 5);

          if (distance < 0.5) distance = 0.5;

          repulsion.x += diff.x / distance;
          repulsion.y += diff.y / distance;

          counter++;
        }
      }
    }
  }

  // TODO
  if (DEBUG && i == n / 2 && counter > 0) {
    printf("distance: %lf; interactions: %d\n", minDistance, counter);
  }

  return repulsion;
}

__global__
void kernelSwarmMove(uchar4 *image, Particle *particles, int n,
                     float2 global_minimum, curandState *state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetx = blockDim.x * gridDim.x;

  // The speed_coeff drops from 5 to 0.01 depending on the time
  float speed_coeff = 5.0 / (1.0 + 0.1 * pow(devTimeValue, 4));
  speed_coeff = speed_coeff < 0.01 ? 0.01 : speed_coeff;

  if (DEBUG && idx == 0) {
    printf("speed_coeff: %lf\n", speed_coeff);
  }

  float rnd_1, rnd_2, rnd_3, rnd_4;
  Particle p;
  float2 repulsion;

  for (int i = idx; i < n; i += offsetx) {
    rnd_1 = curand_uniform(&state[i]);
    rnd_2 = curand_uniform(&state[i]);
    rnd_3 = curand_uniform(&state[i]);
    rnd_4 = curand_uniform(&state[i]);

    p = particles[i];

    p.speed.x = DAMPING_COEFF * p.speed.x +
                 speed_coeff * (
                   rnd_1 * LOCAL_COEFF * (p.best_coords.x - p.coords.x) +
                   rnd_2 * GLOBAL_COEFF * (global_minimum.x - p.coords.x) +
                   RANDOM_COEFF * (rnd_3 - 0.5)
                 );

    p.speed.y = DAMPING_COEFF * p.speed.y +
                 speed_coeff * (
                   rnd_1 * LOCAL_COEFF * (p.best_coords.y - p.coords.y) +
                   rnd_2 * GLOBAL_COEFF * (global_minimum.y - p.coords.y) +
                   RANDOM_COEFF * (rnd_4 - 0.5)
                 );

    repulsion = calculateRepulsionClosest(particles, n, i);
    p.speed.x += REPULSION_COEFF * repulsion.x;
    p.speed.y += REPULSION_COEFF * repulsion.y;

    p.coords.x += p.speed.x;
    p.coords.y += p.speed.y;

    particles[i] = p;
  }
}

void copySizeToGPU() {
  CSC(cudaMemcpyToSymbol((const void *)&devWidth, &width, sizeof(int)));
  CSC(cudaMemcpyToSymbol((const void *)&devHeight, &height, sizeof(int)));
}

void copyZoomToGPU() {
  CSC(cudaMemcpyToSymbol((const void *)&devZoomX, &zoomX, sizeof(float)));
  CSC(cudaMemcpyToSymbol((const void *)&devZoomY, &zoomY, sizeof(float)));
}

void copyParticleSizeToGPU() {
  CSC(cudaMemcpyToSymbol((const void *)&devParticleSize, &particleSize, sizeof(int)));
}

void copyCenterToGPU() {
  CSC(cudaMemcpyToSymbol((const void *)&devCenterX, &centerX, sizeof(float)));
  CSC(cudaMemcpyToSymbol((const void *)&devCenterY, &centerY, sizeof(float)));
}

void copyTimeToGPU() {
  CSC(cudaMemcpyToSymbol((const void *)&devTimeValue, &timeValue, sizeof(float)));
  CSC(cudaMemcpyToSymbol((const void *)&devTimeStep, &timeStep, sizeof(float)));
}

void copyUniformSpaceToGPU(float minX, float maxX, float minY, float maxY, float cellSize) {
  CSC(cudaMemcpyToSymbol((const void *)&devUniformSpaceMinX, &minX, sizeof(float)));
  CSC(cudaMemcpyToSymbol((const void *)&devUniformSpaceMaxX, &maxX, sizeof(float)));
  CSC(cudaMemcpyToSymbol((const void *)&devUniformSpaceMinY, &minY, sizeof(float)));
  CSC(cudaMemcpyToSymbol((const void *)&devUniformSpaceMaxY, &maxY, sizeof(float)));
  CSC(cudaMemcpyToSymbol((const void *)&devUniformSpaceCellSize, &cellSize, sizeof(float)));
}

void copyToGPU() {
  copySizeToGPU();
  copyZoomToGPU();
  copyParticleSizeToGPU();
  copyCenterToGPU();
  copyTimeToGPU();
}

void update() {

  auto t_start = std::chrono::high_resolution_clock::now();

  copyToGPU();

  uchar4 *devHeatMap;
  size_t size;

  CSC(cudaGraphicsMapResources(1, &res, 0));
  CSC(cudaGraphicsResourceGetMappedPointer((void **)&devHeatMap, &size, res));

  // Update the function values and local minima for each particle
  kernelSwarmUpdate<<<blocks_1d, threads_1d>>>(devHeatMap, devParticleArray, numberOfParticles);

  // The boundaries and the center of the swarm, minimum, maximum, the global minimum
  thrust::device_ptr<ParticleArea> startParticleAreaArray(devPartileAreaArray);
  thrust::device_ptr<ParticleArea> endParticleAreaArray = startParticleAreaArray + numberOfParticles;

  ParticleArea pa;
  pa.min_x = FLT_MAX ;
  pa.min_y = FLT_MAX ;
  pa.max_x = -FLT_MAX ;
  pa.max_y = -FLT_MAX ;
  pa.sum_x = 0.0;
  pa.sum_y = 0.0;
  pa.minValue = FLT_MAX ;
  pa.maxValue = -FLT_MAX ;
  pa.globalMinimum = FLT_MAX ;

  kernelInitParticleArea<<<blocks_1d, threads_1d>>>(
    devParticleArray, devPartileAreaArray, numberOfParticles
  );
  pa = thrust::reduce(startParticleAreaArray, endParticleAreaArray, pa, ParticleReductionFunctor());

  // Align the window in the center of particle swarm
  if (autoCenter) {
      centerX = pa.sum_x / numberOfParticles;
      centerY = pa.sum_y / numberOfParticles;
      copyCenterToGPU();
  }

  // Draw a heat map and particles
  kernelNormalizedHeatMap<<<blocks_2d, threads_2d>>>(devHeatMap, pa.minValue, pa.maxValue);
  kernelSwarmDraw<<<blocks_1d, threads_1d>>>(devHeatMap, devParticleArray, numberOfParticles);

  // Space partitioning
  copyUniformSpaceToGPU(pa.min_x, pa.max_x, pa.min_y, pa.max_y, cellSize);
  kernelSwarmAssociateWithCells<<<blocks_1d, threads_1d>>>(devParticleArray, numberOfParticles);

  // Sort particles by cell index
  thrust::device_ptr<Particle> startParticleArray(devParticleArray);
  thrust::device_ptr<Particle> endParticleArray = startParticleArray + numberOfParticles;
  thrust::sort(startParticleArray, endParticleArray, ParticleSortByIndexComparator());

  // Update particles position
  kernelSwarmMove<<<blocks_1d, threads_1d>>>(
    devHeatMap, devParticleArray, numberOfParticles, pa.globalMinimumCoords, devRandomState
  );

  timeValue += timeStep;

  CSC(cudaDeviceSynchronize());
  CSC(cudaGraphicsUnmapResources(1, &res, 0));

  auto t_end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double, std::milli>(t_end-t_start).count();
  if (DEBUG) {
    printf("%lf ms; center: %lf, %lf\n\n", duration, centerX, centerY);
  }

  glutPostRedisplay();
}

void display() {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);
  glutInitWindowSize(width, height);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glutSwapBuffers();
}

void reshapeFunc(int w, int h) {
  width = w;
  height = h;
  zoomY = zoomX * height / width;

  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
  CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));
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
  else if (particleSize > 100) particleSize = 100;
}

void mouseWheelFunc(int wheel, int direction, int x, int y) {
  zoomX += direction < 0 ? 0.1 * zoomX : -0.1 * zoomX;
  if (zoomX < 0.01) zoomX = 0.01;
  else if (zoomX > 1000000) zoomX = 1000000;
  zoomY = zoomX * height / width;
}

int main(int argc, char **argv) {

  std::cout << "Enter window width: ";
  std::cin >> width;

  std::cout << "Enter window height: ";\
  std::cin >> height;

  std::cout << "Enter number of particles: ";
  std::cin >> numberOfParticles;

  std::cout << "Enter cell size: ";
  std::cin >> cellSize;

  copyToGPU();

  cudaMalloc((void **)&devRandomState, sizeof(curandState) * numberOfParticles);
  cudaMalloc((void **)&devParticleArray, sizeof(Particle) * numberOfParticles);
  cudaMalloc((void **)&devPartileAreaArray, sizeof(ParticleArea) * numberOfParticles);

  initRandomState<<<blocks_1d, threads_1d>>>(devRandomState, numberOfParticles);
  kernelSwarmInit<<<blocks_1d, threads_1d>>>(devParticleArray, numberOfParticles, devRandomState);

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(width, height);
  glutCreateWindow("Particle swarm optimization");

  glutIdleFunc(update);
  glutDisplayFunc(display);
  glutReshapeFunc(reshapeFunc);
  glutKeyboardFunc(keyboardFunc);
  glutMouseWheelFunc(mouseWheelFunc);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);

  glewInit();
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);

  reshapeFunc(width, height);

  glutMainLoop();

  CSC(cudaGraphicsUnregisterResource(res));

  glBindBuffer(1, vbo);
  glDeleteBuffers(1, &vbo);

  cudaFree(devRandomState);
  cudaFree(devParticleArray);
  cudaFree(devRandomState);

  return 0;
}
