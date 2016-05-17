/*
Particle swarm optimization
by Ivan Vinogradov
2016
*/

#include <iostream>
#include <algorithm>
#include <random>
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

#include "common.cuh"
#include "particle.cuh"

// Window size, center, zoom, function min-max, time
double devFuncMin;
double devFuncMax;
double devTimeValue;
double devTimeStep;

// Parameters of uniform space partitioning
double devUniformSpaceMinX;
double devUniformSpaceMaxX;
double devUniformSpaceMinY;
double devUniformSpaceMaxY;
double devUniformSpaceCellSize;

uchar4 *heatMap;
Particle *particles;
std::mt19937 randomEngine;
std::uniform_real_distribution<double> randomDoubleDistribution;

double randomDouble() {
    return randomDoubleDistribution(randomEngine);
}

// Screen coordinates into real coordinates
double2 indexToCoord(int2 index) {
  return make_double2(
    (2.0f * index.x / (double)(width - 1) - 1.0f) * zoomX + centerX,
    -(2.0f * index.y / (double)(height - 1) - 1.0f) * zoomY + centerY);
}

// Real coordinates into screen coordinates
int2 coordToIndex(double2 coord) {
  return make_int2(
    0.5f * (width - 1) * (1.0f + (coord.x - centerX) / zoomX),
    0.5f * (height - 1) * (1.0f - (coord.y - centerY) / zoomY)
  );
}

// Schwefel Function
double fun(double2 coord) {
  return -coord.x * sin(sqrt(fabs(coord.x))) - coord.y * sin(sqrt(fabs(coord.y)));
}

double fun(int2 index) {
  return fun(indexToCoord(index));
}

void kernelSwarmInit(Particle *particles, int n) {
  Particle *p;
  for (int i = 0; i < n; i ++) {
    p = &particles[i];

    // Position in the center of the screen
    p->coords = p->best_coords = indexToCoord(make_int2(width / 2, height / 2));

    // // Random position within the screen
    // p->coords = p->best_coords = indexToCoord(make_int2(
    //   randomDouble() * width, randomDouble() * height
    // ));

    // Random starting angle and the speed
    double angle = 2.0 * 3.14 * randomDouble();
    double speed = 100.0 * randomDouble();
    p->speed = make_double2(cos(angle) * speed, sin(angle) * speed);
    p->value = p->best_value = DBL_MAX;
  }
}

void kernelSwarmUpdate(uchar4 *image, Particle *particles, int n) {
  Particle *p;
  for (int i = 0; i < n; i ++) {
    p = &particles[i];
    p->value = fun(p->coords);
    if (p->value < p->best_value) {
      p->best_value = p->value;
      p->best_coords = p->coords;
    }
  }
}

void kernelNormalizedHeatMap(uchar4 *heatMap, double minValue, double maxValue) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      double f = (fun(make_int2(i, j)) - minValue) / (maxValue - minValue);
      if (f < 0.0) f = 0.0; else if (f > 1.0) f = 1.0;
      heatMap[j * width + i] = make_uchar4(
        (int)(f * 255), 0, (int)((1.0 - f) * 255), 255
      );
    }
  }
}

void kernelSwarmDraw(uchar4 *heatMap, Particle *particles, int n) {
  int size = particleSize;
  for (int i = 0; i < n; i++) {
    int2 index = coordToIndex(particles[i].coords);
    for (int x = index.x - size; x <= index.x + size; x++) {
      for (int y = index.y - size; y <= index.y + size; y++) {
        if (x >= 0 && x < width && y >= 0 && y < height &&
            (x - index.x) * (x - index.x) + (y - index.y) * (y - index.y) <= size * size
        ) {

          if (DEBUG && i == n / 2) {
            heatMap[y * width + x] = make_uchar4(0, 255, 0, 255);
            continue;
          }

          heatMap[y * width + x] = make_uchar4(255, 255, 255, 255);
        }
      }
    }
  }
}

// The assignment for each particle corresponding space partitioning cell
void kernelSwarmAssociateWithCells(Particle *particles, int n) {
  int sizeX = ceil((devUniformSpaceMaxX - devUniformSpaceMinX) / devUniformSpaceCellSize);
  int sizeY = ceil((devUniformSpaceMaxY - devUniformSpaceMinY) / devUniformSpaceCellSize);

  Particle *p;
  int cellX, cellY;

  for (int i = 0; i < n; i++) {
    p = &particles[i];
    cellX = (p->coords.x - devUniformSpaceMinX) / devUniformSpaceCellSize;
    cellY = (p->coords.y - devUniformSpaceMinY) / devUniformSpaceCellSize;
    p->cellIndex = cellX * sizeX + cellY;
  }
}

// The total force of repulsion for the i-th particle (without the space partitioning)
double2 calculateRepulsionAll(Particle *particles, int n, int i) {
  double2 repulsion, diff, coords_a, coords_b;
  double distance;

  // TODO
  double minDistance = DBL_MAX;

  repulsion.x = 0.0;
  repulsion.y = 0.0;
  coords_a = (&particles[i])->coords;

  for (int j = 0; j < n; j++) {
    if (j == i) continue;
    coords_b = (&particles[j])->coords;
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

  // TODO
  // repulsion.x /= n;
  // repulsion.y /= n;
  return repulsion;
}

// Binary search in a sorted array of particles by cell index of space partitioning
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
int isFound(Particle *particles, int size, int cellIndex, int index) {
    return index < size && particles[index].cellIndex == cellIndex;
}

// Search the first particle with specified cell partition index
int findParticleByCell(Particle *particles, int size, int cellIndex) {
  int index = binarySearchLowerBound(particles, size, cellIndex);
  return isFound(particles, size, cellIndex, index) ? index : -1;
}

// The total force of repulsion for the i-th particle (with the space partitioning)
double2 calculateRepulsionClosest(Particle *particles, int n, int i) {
  double2 diff, repulsion = make_double2(0.0, 0.0);
  double distance;

  // TODO
  double minDistance = DBL_MAX;

  // Counter of interacting particles
  int counter = 0;

  // Dimensions of the space partitioning
  int sizeX = ceil((devUniformSpaceMaxX - devUniformSpaceMinX) / devUniformSpaceCellSize);
  int sizeY = ceil((devUniformSpaceMaxY - devUniformSpaceMinY) / devUniformSpaceCellSize);

  if (sizeX < 1) sizeX = 1;
  if (sizeY < 1) sizeY = 1;

  Particle *pa, *pb;

  pa = &particles[i];

  int cellIndex = pa->cellIndex;
  int cellX = cellIndex / sizeX;
  int cellY = cellIndex % sizeX;

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
          pb = &particles[k];
          if (pb->cellIndex != neighborCellIndex) break;

          diff.x = pa->coords.x - pb->coords.x;
          diff.y = pa->coords.y - pb->coords.y;

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

  // TODO
  // if (counter > 1) {
  //   repulsion.x /= counter;
  //   repulsion.y /= counter;
  // }

  return repulsion;
}

void kernelSwarmMove(uchar4 *image, Particle *particles, int n, double2 global_minimum) {
  const double g_coeff = 0.000050;     // coefficient of global solution
  const double p_coeff = 0.00000010;   // coefficient of local solution
  const double rnd_coeff = 0.010;      // coefficient of random motion
  const double damping_coeff = 0.991;  // coefficient of damping force
  const double repulsion_coeff = 0.10; // coefficient of repulsive force

  // The speed drops to 0.1 depending on the time
  double speed_coeff = 1.0 / (1.0 + 0.01 * pow(devTimeValue, 4));
  speed_coeff = speed_coeff < 0.1 ? 0.1 : speed_coeff;

  double rnd_1, rnd_2, rnd_3, rnd_4;
  Particle *p;
  double2 repulsion;

  for (int i = 0; i < n; i++) {
    rnd_1 = randomDouble();
    rnd_2 = randomDouble();
    rnd_3 = randomDouble();
    rnd_4 = randomDouble();

    p = &particles[i];

    p->speed.x = damping_coeff * p->speed.x +
                 speed_coeff * (
                   rnd_1 * p_coeff * (p->best_coords.x - p->coords.x) +
                   rnd_2 * g_coeff * (global_minimum.x - p->coords.x) +
                   rnd_coeff * (rnd_3 - 0.5)
                 );

    p->speed.y = damping_coeff * p->speed.y +
                 speed_coeff * (
                   rnd_1 * p_coeff * (p->best_coords.y - p->coords.y) +
                   rnd_2 * g_coeff * (global_minimum.y - p->coords.y) +
                   rnd_coeff * (rnd_4 - 0.5)
                 );

    // repulsion = calculateRepulsionAll(particles, n, i);
    repulsion = calculateRepulsionClosest(particles, n, i);
    p->speed.x += repulsion_coeff * repulsion.x;
    p->speed.y += repulsion_coeff * repulsion.y;

    p->coords.x += p->speed.x;
    p->coords.y += p->speed.y;
  }
}

void update() {

  auto t_start = std::chrono::high_resolution_clock::now();

  // Update the function values and local minima for each particle
  kernelSwarmUpdate(heatMap, particles, numberOfParticles);

  // The boundaries and the center of the swarm, minimum, maximum, the global minimum
  ParticleArea result;
  result.min_x = DBL_MAX;
  result.min_y = DBL_MAX;
  result.max_x = -DBL_MAX;
  result.max_y = -DBL_MAX;
  result.sum_x = 0.0;
  result.sum_y = 0.0;
  result.minValue = DBL_MAX;
  result.maxValue = -DBL_MAX;
  result.globalMinimum = DBL_MAX;

  for (int i = 0; i < numberOfParticles; i++) {
    Particle p = particles[i];

    if (p.coords.x < result.min_x) result.min_x = p.coords.x;
    if (p.coords.x > result.max_x) result.max_x = p.coords.x;

    if (p.coords.y < result.min_y) result.min_y = p.coords.y;
    if (p.coords.y > result.max_y) result.max_y = p.coords.y;

    if (p.value < result.minValue) result.minValue = p.value;
    if (p.value > result.maxValue) result.maxValue = p.value;

    result.sum_x += p.coords.x;
    result.sum_y += p.coords.y;

    if (p.best_value < result.globalMinimum) {
      result.globalMinimum = p.best_value;
      result.globalMinimumCoords = p.best_coords;
    }
  }

  // Align the window in the center of particle swarm
  if (autoCenter) {
    centerX = result.sum_x / numberOfParticles;
    centerY = result.sum_y / numberOfParticles;
  }

  // Draw a heat map and particles
  kernelNormalizedHeatMap(heatMap, result.minValue, result.maxValue);
  kernelSwarmDraw(heatMap, particles, numberOfParticles);

  // Space partitioning
  devUniformSpaceMinX = result.min_x;
  devUniformSpaceMaxX = result.max_x;
  devUniformSpaceMinY = result.min_y;
  devUniformSpaceMaxY = result.max_y;
  devUniformSpaceCellSize = cellSize;

  kernelSwarmAssociateWithCells(particles, numberOfParticles);

  // Sort particles by cell index
  std::sort(particles, particles + numberOfParticles, ParticleSortByIndexComparator());

  // Update particles position
  kernelSwarmMove(heatMap, particles, numberOfParticles, result.globalMinimumCoords);

  timeValue += timeStep;

  auto t_end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double, std::milli>(t_end-t_start).count();
  if (DEBUG) {
    printf("%lf ms; center: %lf, %lf\n\n", duration, centerX, centerY);
  }

  glutPostRedisplay();
}

void display() {
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, heatMap);
  glutSwapBuffers();
}

void reshapeFuncCPU(int w, int h) {
  width = w;
  height = h;
  zoomY = zoomX * height / width;

  delete[] heatMap;
  heatMap = new uchar4[width * height];

  glutInitWindowSize(width, height);
  glutPostRedisplay();
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

  particles = new Particle[numberOfParticles];
  heatMap = new uchar4[width * height];

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  randomEngine = std::mt19937(seed);
  randomDoubleDistribution = std::uniform_real_distribution<double>(0.0, 1.0);

  kernelSwarmInit(particles, numberOfParticles);

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(width, height);
  glutCreateWindow("Particle swarm optimization");

  glutIdleFunc(update);
  glutDisplayFunc(display);
  glutReshapeFunc(reshapeFuncCPU);
  glutKeyboardFunc(keyboardFunc);
  glutMouseWheelFunc(mouseWheelFunc);

  glutMainLoop();

  delete[] particles;
  delete[] heatMap;

  return 0;
}
