#include "gpumanager.h"

//The cluster initizalization of kmeans
__global__ void kMeansInit(float *redBuffer, const float *greenBuffer, const float *blueBuffer, const unsigned int size, const int numOfClusters, float *centroids,
                            float *newCentroids, int *centroidsQuantity, const unsigned int *randomSeeds)
{
    int globalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

    const int DIM = 3;
    
    const int indX = globalIdx * DIM;
    const int indY = indX + 1;
    const int indZ = indY + 1;
    
    int random;
    
    // choose more centers
    if (globalIdx < numOfClusters) {

        // Getting a random index within each size / number of clusters wide interval
        random = globalIdx * (size / numOfClusters) + (randomSeeds[globalIdx] % (size / numOfClusters));

        centroids[indX] = redBuffer[random];
        centroids[indY] = greenBuffer[random];
        centroids[indZ] = blueBuffer[random];
    }
        
    centroidsQuantity[globalIdx] = 0;
        
    newCentroids[globalIdx] = 0;
}

// One iteration of kmeans
__global__ void kMeansIteration(float *redBuffer, const float *greenBuffer, const float *blueBuffer, unsigned short *labels, const unsigned int size, const int numOfClusters,
                         float *centroids)
{
    int globalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (globalIdx >= size)
        return;

    const int DIM = 3;

    int localIdx = threadIdx.x;

    __shared__ float centroidsLocal[DIM * numOfClusters];


    // Copying the centroids of the clusters to shared memory for increased speed
    if (localIdx < DIM * numOfClusters) {
        centroidsLocal[localIdx] = centroids[localIdx];
    }

    __syncthreads();
    
    float distance, newDistance, x, y, z;
    int centroidsIdx = 0;

    x = redBuffer[globalIdx] - centroidsLocal[0];
    y = greenBuffer[globalIdx] - centroidsLocal[1];
    z = blueBuffer[globalIdx] - centroidsLocal[2];

    distance = x * x + y * y + z * z;
    
    // look for a smaller distance in the rest of centroids
    for (int j = 1; j < numOfClusters; j++) {
        
        x = redBuffer[globalIdx] - centroidsLocal[j * DIM];
        y = greenBuffer[globalIdx] - centroidsLocal[j * DIM + 1];
        z = blueBuffer[globalIdx] - centroidsLocal[j * DIM + 2];

        newDistance = x * x + y * y + z * z;
        
        if (newDistance < distance) {

            centroidsIdx = j;

            distance = newDistance;
        }
    }
    
    labels[globalIdx] = centroidsIdx;
}

// The original kmeans algorithm
__global__ void kMeans(float *redBuffer, const float *greenBuffer, const float *blueBuffer, unsigned short *labels, const unsigned int size, const int numOfClusters,
                         float *centroids, float *newCentroids, int *centroidsQuantity,  float *distanceAccumulation, const unsigned int *randomSeeds)
{
    // get index into global data array
    int globalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (globalIdx >= size)
        return;
    
    // num of dimensions
    const int DIM = 3;
    
    const int indX = globalIdx * DIM;
    const int indY = indX + 1;
    const int indZ = indY + 1;
    
    float *tempPtr;
    float distance, newDistance, x, y, z;
    int random;
    
    // choose more centers
    if (globalIdx < numOfClusters) {

        // Getting a random index within each size / number of clusters wide interval
        random = globalIdx * (size / numOfClusters) + (randomSeeds[globalIdx] % (size / numOfClusters));

        centroids[indX] = redBuffer[random];
        centroids[indY] = greenBuffer[random];
        centroids[indZ] = blueBuffer[random];
    }

    
    //////////////////////////////////////////////////////////////////////////
    // Synchronize to make sure all threads are done
    __syncthreads();
    //////////////////////////////////////////////////////////////////////////
    
    const float epsilon = 1e-4f;
    int centroidsIdx;
    
    int numOfIterations = 0;
    
    // Until 5 iterations
    while (numOfIterations < 5) {
        
        // Empty all clusters before classification
        if (globalIdx >= 0 && globalIdx < numOfClusters) {
            
            centroidsQuantity[globalIdx] = 0;
        }
        
        if (globalIdx >= 0 && globalIdx < numOfClusters * DIM) {
            
            newCentroids[globalIdx] = 0;
        }
        
        //////////////////////////////////////////////////////////////////////////
        // Synchronize to make sure all threads are done
        __syncthreads();
        //////////////////////////////////////////////////////////////////////////
        
        // Use the estimated means to classify the samples into K clusters
        // estimate the distance between colors[Idx] and centroids[0]
        centroidsIdx = 0;

        x = redBuffer[globalIdx] - centroids[0];
        y = greenBuffer[globalIdx] - centroids[1];
        z = blueBuffer[globalIdx] - centroids[2];

        distance = x * x + y * y + z * z;
        
        // look for a smaller distance in the rest of centroids
        for (int j = 1; j < numOfClusters; j++) {
            
            x = redBuffer[globalIdx] - centroids[j * DIM];
            y = greenBuffer[globalIdx] - centroids[j * DIM + 1];
            z = blueBuffer[globalIdx] - centroids[j * DIM + 2];

            newDistance = x * x + y * y + z * z;
            
            if (newDistance < distance) {
    
                centroidsIdx = j;

                distance = newDistance;
            }
        }
        
        labels[globalIdx] = centroidsIdx;
        centroidsQuantity[centroidsIdx]++;
        newCentroids[centroidsIdx * DIM] += redBuffer[globalIdx];
        newCentroids[centroidsIdx * DIM + 1] += greenBuffer[globalIdx];
        newCentroids[centroidsIdx * DIM + 2] += blueBuffer[globalIdx];
        
        ////////////////////////////////////////////////////////////////////////
        //Synchronize to make sure all threads are done
       __syncthreads();
        ////////////////////////////////////////////////////////////////////////
        
        if (globalIdx >= 0 && globalIdx < numOfClusters) {
            
            // estimate the values of the new centers
            if (centroidsQuantity[globalIdx] > 0) {
                
                newCentroids[indX] /= centroidsQuantity[globalIdx];
                newCentroids[indY] /= centroidsQuantity[globalIdx];
                newCentroids[indZ] /= centroidsQuantity[globalIdx];
            }
            
            newDistance
            = fabs(centroids[indX] - newCentroids[indX])
            + fabs(centroids[indY] - newCentroids[indY])
            + fabs(centroids[indZ] - newCentroids[indZ]);
            
            if (distance > epsilon) {
                
                distanceAccumulation[globalIdx] += 1;
            }
            
       }
        
        ////////////////////////////////////////////////////////////////////////
        //Synchronize to make sure all threads are done
        __syncthreads();
        ////////////////////////////////////////////////////////////////////////
        
        
        if (globalIdx == 0) {
            
            // swap the new centroids with the old ones
            tempPtr = centroids;
            centroids = newCentroids;
            newCentroids = tempPtr;
        }

       ++numOfIterations;
       __syncthreads();
    }
}


// Replacing the colors with the center of their cluster
__global__ void convertToKRangeColors(float *redBuffer, float *greenBuffer, float *blueBuffer, const unsigned long int size, float *centroids, unsigned short *labels)
{
    int globalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if(globalIdx >= size)
        return;
    
    redBuffer[globalIdx] = centroids[labels[globalIdx] * 3];
    greenBuffer[globalIdx] = centroids[labels[globalIdx] * 3 + 1];
    blueBuffer[globalIdx] = centroids[labels[globalIdx] * 3 + 2];
}


GpuManager::GpuManager() : numOfClusters(256)
{
}


// Cleaning memory on GPU
void GpuManager::clearBuffers()
{
    cudaFree(this->dRedBuffer);
    cudaFree(this->dGreenBuffer);
    cudaFree(this->dBlueBuffer);

    cudaFree(this->centroids);
    cudaFree(this->newCentroids);
    cudaFree(this->centroidsQuantity);

    cudaFree(this->distanceAccumulation);
    cudaFree(this->dSeeds);
    cudaFree(this->labels);
}


// Allocating memory on GPU
void GpuManager::initBuffers()
{
    cudaMalloc((void **) &this->dRedBuffer, this->size * sizeof(float));
    cudaMalloc((void **) &this->dGreenBuffer, this->size * sizeof(float));
    cudaMalloc((void **) &this->dBlueBuffer, this->size * sizeof(float));

    cudaMalloc((void **) &this->centroids, 3 * this->numOfClusters * sizeof(float));
    cudaMalloc((void **) &this->newCentroids, 3 * this->numOfClusters * sizeof(float));
    cudaMalloc((void **) &this->centroidsQuantity, this->numOfClusters * sizeof(int));

    cudaMalloc((void **) &this->distanceAccumulation, this->size * sizeof(float));
    cudaMalloc((void **) &this->dSeeds, this->numOfClusters * sizeof(unsigned int));
    cudaMalloc((void **) &this->labels, this->size * sizeof(unsigned int));
}


void GpuManager::setSize(const unsigned int inputSize)
{
    this->size = inputSize;
}


// Copy the color data from host memory to device memory
void GpuManager::copyDataImageToDevice(float *redBuffer, float *greenBuffer, float *blueBuffer)
{
    cudaMemcpy(this->dRedBuffer, redBuffer, this->size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(this->dGreenBuffer, greenBuffer, this->size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(this->dBlueBuffer, blueBuffer, this->size * sizeof(float), cudaMemcpyHostToDevice);
}


//Generate and copy random numbers for initial seeds to device memory
void GpuManager::generateAndCopyRandomSeedsToDevice()
{
    unsigned int *seeds == new unsigned int[this->gpuManager.getNumOfClusters()];

    srand( (unsigned)time( NULL ) );

    for (int j = 0; j < this->gpuManager.getNumOfClusters(); ++j) {
            seeds[j] = rand();
    }

    cudaMemcpy(this->dSeeds, seeds, this->numOfClusters * sizeof(unsigned int), cudaMemcpyHostToDevice);

    delete[] seeds;
}


// Copy the color data from device memory to host memory
void GpuManager::copyDataFromDevice(float *redBuffer, float *greenBuffer, float *blueBuffer)
{
    cudaMemcpy(redBuffer, this->dRedBuffer, this->size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(greenBuffer, this->dGreenBuffer, this->size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(blueBuffer, this->dBlueBuffer, this->size * sizeof(float), cudaMemcpyDeviceToHost);
}


// Copy the colorpalette and the clusterindex of each pixel to host memory
void GpuManager::copyImageDataFromDevice(float *palette, unsigned short *indexes)
{
    cudaMemcpy(palette, this->centroids, 3 * this->numOfClusters * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(indexes, this->labels, this->size * sizeof(unsigned short), cudaMemcpyDeviceToHost);
}


// Executing KMeans and Clustering kernel algorithms
void GpuManager::quantizeImage()
{
    // The number of threads within a block (hardware limit is 512 on old GPUs, 1024 on new)
    const dim3 blockSize(1024);

    // The number of blocks, each will launch a 1024 threads
    const dim3 gridSize(this->size/1024 + 1);

    kMeansInit<<<8, 32>>>(this->dRedBuffer, this->dGreenBuffer, this->dBlueBuffer, this->size, this->numOfClusters, this->centroids, this->dSeeds);

    // kMeans<<<gridSize, blockSize>>>(this->dRedBuffer, this->dGreenBuffer, this->dBlueBuffer, this->labels, this->size, this->numOfClusters,
    //     this->centroids, this->newCentroids, this->centroidsQuantity,  this->distanceAccumulation, this->dSeeds);

    cudaDeviceSynchronize();

    kMeansIteration<<<gridSize, blockSize>>>(this->dRedBuffer, this->dGreenBuffer, this->dBlueBuffer, this->labels, this->size, this->numOfClusters,
                                 this->centroids, this->newCentroids, this->centroidsQuantity);

    cudaDeviceSynchronize();

    //convertToKRangeColors<<<gridSize, blockSize>>>(this->dRedBuffer, this->dGreenBuffer, this->dBlueBuffer, this->size, this->centroids, this->labels);
}


int GpuManager::getNumOfClusters()
{
    return this->numOfClusters;
}
