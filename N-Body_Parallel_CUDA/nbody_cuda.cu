// This code simulates the gravitational interaction between multiple bodies in a 3D space using CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>
#include <string>
#include <fstream>
#include <cstdlib>  // For rand()
#include <chrono> // For tracking runtime

#define G 6.67430e-8    // Gravitational constant
#define DT 0.1          // Time step
#define NUM_BODIES 100  // Number of bodies (only change this number to scale up the computation)
#define NUM_STEPS 5000  // Number of simulation steps (5000)
#define WIDTH 1000.0    // Visualization width
#define HEIGHT 1000.0   // Visualization height
#define DEPTH 1000.0    // Visualization depth
#define VELOCITY_RANGE 100  // Positive and Negative range for random velocity
#define MAX_MASS 1000000     // Range for random mass value


float bodies[NUM_BODIES][7]; // 0=x, 1=y, 2=z, 3=vx, 4=vy, 5=vz, 6=mass
float forces[NUM_BODIES][3]; // 0=fx, 1=fy, 2=fz,


__global__ void kernelComputeForcesAndUpdatePositions2D(float* deviceBodies, float* deviceForces,
    int numBodies, float g, float dt,
    float width, float height, float depth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Body i
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Body j

    if (i >= numBodies || j >= numBodies || i == j) return;

    // Compute the force of body j on body i
    int curBodyIndex = i * 7; // index for the current body
    int compareIndex = j * 7; // index for the current body to calculate forces against
    int curForceIndex = i * 3; // index for the current body forces

    // Calculate the forces based on the distance and mass between each combination of body i and all other j bodies
    float distX = deviceBodies[compareIndex + 0] - deviceBodies[curBodyIndex + 0];
    float distY = deviceBodies[compareIndex + 1] - deviceBodies[curBodyIndex + 1];
    float distZ = deviceBodies[compareIndex + 2] - deviceBodies[curBodyIndex + 2];

    float distance = sqrt(distX * distX + distY * distY + distZ * distZ);
    float force = g * deviceBodies[curBodyIndex + 6] * deviceBodies[compareIndex + 6] / (distance * distance);

    // Calculate individual axis forces
    float fx = force * (distX / distance);
    float fy = force * (distY / distance);
    float fz = force * (distZ / distance);

    // Use atomic operations to accumulate forces
    atomicAdd(&deviceForces[curForceIndex + 0], fx);
    atomicAdd(&deviceForces[curForceIndex + 1], fy);
    atomicAdd(&deviceForces[curForceIndex + 2], fz);

    // Update positions and velocities (only one thread per body should do this)
    if (j == 0) {
        deviceBodies[curBodyIndex + 3] += deviceForces[curForceIndex + 0] / deviceBodies[curBodyIndex + 6] * dt; // body VX += forceX / mass * dt
        deviceBodies[curBodyIndex + 4] += deviceForces[curForceIndex + 1] / deviceBodies[curBodyIndex + 6] * dt;
        deviceBodies[curBodyIndex + 5] += deviceForces[curForceIndex + 2] / deviceBodies[curBodyIndex + 6] * dt;

        deviceBodies[curBodyIndex + 0] += deviceBodies[curBodyIndex + 3] * dt; // body X += VX * dt
        deviceBodies[curBodyIndex + 1] += deviceBodies[curBodyIndex + 4] * dt;
        deviceBodies[curBodyIndex + 2] += deviceBodies[curBodyIndex + 5] * dt;

        // If a body hits a boundary, reverse it's velocity
        if (deviceBodies[curBodyIndex + 0] >= width || deviceBodies[curBodyIndex + 0] <= 0) {
            deviceBodies[curBodyIndex + 3] = -deviceBodies[curBodyIndex + 3];
        }
        if (deviceBodies[curBodyIndex + 1] >= height || deviceBodies[curBodyIndex + 1] <= 0) {
            deviceBodies[curBodyIndex + 4] = -deviceBodies[curBodyIndex + 4];
        }
        if (deviceBodies[curBodyIndex + 2] >= depth || deviceBodies[curBodyIndex + 2] <= 0) {
            deviceBodies[curBodyIndex + 5] = -deviceBodies[curBodyIndex + 5];
        }
    }
}


void save_to_csv(std::ofstream& file, int step, float*bodies_this_step) {
    for (int i = 0; i < NUM_BODIES; i++) {
		int index = i * 7; // Calculate the index for the current body
        file << step << "," << i << "," << bodies_this_step[index] << "," << bodies_this_step[index+1] << "," << bodies_this_step[index+2] << "\n";
    }
}


void run_simulation() {
    std::ofstream file("nbody_output.csv");
    file << "step,id,x,y,z\n";


    // Allocate pinned host memory for bodies
    float* pinnedHostBodies = nullptr; // Use a single pointer for 2D data
    int bodiesDataSize = sizeof(float) * NUM_BODIES * 7; // Total size of the 2D array
    cudaMallocHost(&pinnedHostBodies, bodiesDataSize); // Allocate pinned host memory
    memcpy(pinnedHostBodies, bodies, bodiesDataSize); // Copy data from the original 2D array to pinned host memory
    
    float* deviceBodies = nullptr; // Allocate device memory
    cudaMalloc(&deviceBodies, bodiesDataSize);
    cudaMemcpy(deviceBodies, pinnedHostBodies, bodiesDataSize, cudaMemcpyHostToDevice); // Copy from pinned host memory to device memory

    // Allocate pinned host memory for forces
    float* pinnedHostForces = nullptr; // Use a single pointer for 2D data
    int forcesDataSize = sizeof(float) * NUM_BODIES * 3; // Total size of the 2D array
    cudaMallocHost(&pinnedHostForces, forcesDataSize); // Allocate pinned host memory
    memcpy(pinnedHostForces, forces, forcesDataSize); // Copy data from the original 2D array to pinned host memory

    float* deviceForces = nullptr; // Allocate device memory
    cudaMalloc(&deviceForces, forcesDataSize);
    cudaMemcpy(deviceForces, pinnedHostForces, forcesDataSize, cudaMemcpyHostToDevice); // Copy from pinned host memory to device memory

    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((NUM_BODIES + blockDim.x - 1) / blockDim.x,
                 (NUM_BODIES + blockDim.y - 1) / blockDim.y);

    for (int step = 0; step < NUM_STEPS; step++) {

        kernelComputeForcesAndUpdatePositions2D <<<gridDim, blockDim >>> (deviceBodies, deviceForces,
                                                                            NUM_BODIES, G, DT, WIDTH, HEIGHT, DEPTH);

        cudaDeviceSynchronize();

		// Copy bodies back from device to pinned host memory
        cudaMemcpy(pinnedHostBodies, deviceBodies, bodiesDataSize, cudaMemcpyDeviceToHost);

        save_to_csv(file, step, pinnedHostBodies);
    }

    file.close();

    // Copy data back from device to pinned host memory
    cudaMemcpy(pinnedHostBodies, deviceBodies, bodiesDataSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(pinnedHostForces, deviceForces, forcesDataSize, cudaMemcpyDeviceToHost);

	// Copy data back from pinned host memory to vectors
    memcpy(bodies, pinnedHostBodies, bodiesDataSize);
    memcpy(forces, pinnedHostForces, forcesDataSize);
}



void initialize_bodies() {
    // Initialize NUM_BODIES particles
    srand(time(0)); // Ensures different random number on each run

    //#pragma omp parallel for schedule(static) //num_threads(8)
    for (int i = 0; i < NUM_BODIES; i++) {
        bodies[i][0] = rand() % int(WIDTH);  // Random num in range 0 - WIDTH
        bodies[i][1] = rand() % int(HEIGHT); // 0=x, 1=y, 2=z, 3=vx, 4=vy, 5=vz, 6=mass
        bodies[i][2] = rand() % int(DEPTH);

        bodies[i][3] = (rand() % (2 * VELOCITY_RANGE)) - VELOCITY_RANGE;  // Random num in range -VELOCITY_RANGE to VELOCITY_RANGE
        bodies[i][4] = (rand() % (2 * VELOCITY_RANGE)) - VELOCITY_RANGE;
        bodies[i][5] = (rand() % (2 * VELOCITY_RANGE)) - VELOCITY_RANGE;

        bodies[i][6] = (rand() % int(MAX_MASS)) + 1;  // Random num in range 1 - (MAX_MASS + 1)
    }
}



int main(int argc, char* argv[]) {
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    initialize_bodies();
    //print_bodies(); // For testing
    run_simulation();

    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print runtime
    std::cout << "\nNumber of bodies: " << NUM_BODIES;
    std::cout << "\nCUDA runtime: " << elapsed.count() << " seconds\n";

    std::cout << "Simulation complete. Data saved to nbody_output.csv\n\n";
    return 0;
}

