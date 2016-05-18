/*
Particle swarm optimization
by Ivan Vinogradov
2016
*/

struct Particle {
  float2 coords;
  float2 best_coords;
  float2 speed;
  float value;
  float best_value;
  int cellIndex;
};

struct ParticleArea {
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float sum_x;
  float sum_y;
  float minValue;
  float maxValue;
  float globalMinimum;
  float2 globalMinimumCoords;
};

/* Thrust sorting comparator */
struct ParticleSortByIndexComparator
{
  __host__ __device__ bool operator()(const Particle &a, const Particle &b) const {
    return a.cellIndex < b.cellIndex;
  }
};

/* Thrust reduction functor */
struct ParticleReductionFunctor {
  __device__ ParticleArea operator()(const ParticleArea &a, const ParticleArea &b) const {
    ParticleArea result;

    result.min_x = a.min_x < b.min_x ? a.min_x : b.min_x;
    result.min_y = a.min_y < b.min_y ? a.min_y : b.min_y;
    result.max_x = a.max_x > b.max_x ? a.max_x : b.max_x;
    result.max_y = a.max_y > b.max_y ? a.max_y : b.max_y;
    result.sum_x = a.sum_x + b.sum_x;
    result.sum_y = a.sum_y + b.sum_y;
    result.minValue = a.minValue < b.minValue ? a.minValue : b.minValue;
    result.maxValue = a.maxValue > b.maxValue ? a.maxValue : b.maxValue;

    if (a.globalMinimum < b.globalMinimum) {
      result.globalMinimum = a.globalMinimum;
      result.globalMinimumCoords = a.globalMinimumCoords;
    } else {
      result.globalMinimum = b.globalMinimum;
      result.globalMinimumCoords = b.globalMinimumCoords;
    }

    return result;
  }
};

__global__
void kernelInitParticleArea(Particle *particleArray, ParticleArea *particleAreaArray, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetx = blockDim.x * gridDim.x;
  Particle p;
  ParticleArea pa;
  for (int i = idx; i < size; i += offsetx) {
    p = particleArray[i];
    pa = particleAreaArray[i];

    pa.min_x = p.coords.x;
    pa.min_y = p.coords.y;
    pa.max_x = p.coords.x;
    pa.max_y = p.coords.y;
    pa.sum_x = p.coords.x;
    pa.sum_y = p.coords.y;
    pa.minValue = p.value;
    pa.maxValue = p.value;
    pa.globalMinimum = p.best_value;
    pa.globalMinimumCoords = p.best_coords;

    particleAreaArray[i] = pa;
  }
}
