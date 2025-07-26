// This code simulates the gravitational interaction between multiple bodies in a 3D space using OpenMP.


#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>
#include <string>
#include <fstream>
#include <cstdlib>  // For rand()
#include <chrono> // For tracking runtime
#include <omp.h>

#define G 6.67430e-8    // Gravitational constant
#define DT 0.1          // Time step
#define NUM_BODIES 100  // Number of bodies (only change this number to scale up the computation)
#define NUM_STEPS 5000  // Number of simulation steps
#define WIDTH 1000.0    // Visualization width
#define HEIGHT 1000.0   // Visualization height
#define DEPTH 1000.0    // Visualization depth
#define VELOCITY_RANGE 100  // Positive and Negative range for random velocity
#define MAX_MASS 1000000     // Range for random mass value

struct Body {
    double x, y, z, vx, vy, vz, mass;
};

std::vector<Body> bodies(NUM_BODIES);
std::vector<std::tuple<double, double, double>> forces(NUM_BODIES);

void compute_forces() {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NUM_BODIES; i++) {

        double fx = 0.0, fy = 0.0, fz = 0.0;

        for (int j = 0; j < NUM_BODIES; j++) {
            if (i != j) {
                //printf("Thread: %d calculating body: %d against body: %d\n", omp_get_thread_num(), i, j); //For testing

                // Calculate the forces based on the distance and mass between each combination of body i and all other bodies
                double distX = bodies[j].x - bodies[i].x;
                double distY = bodies[j].y - bodies[i].y;
                double distZ = bodies[j].z - bodies[i].z;

                double distance = std::sqrt( pow(distX, 2) + pow(distY, 2) + pow(distZ, 2) );

                double force = G * bodies[i].mass * bodies[j].mass / distance;

                // Seperate force into X, Y, Z
                fx += force * (distX / distance);
                fy += force * (distY / distance);
                fz += force * (distZ / distance);
            }
        }
        //printf("\n"); //For testing
        forces[i] = {fx, fy, fz};
    }
}


void update_positions() {
    for (int i = 0; i < NUM_BODIES; i++) {
        // Update velocities based on forces (Hint: this step involves forces[i], bodies[i].mass, and DT)
        bodies[i].vx += std::get<0>(forces[i]) / bodies[i].mass * DT;
        bodies[i].vy += std::get<1>(forces[i]) / bodies[i].mass * DT;
        bodies[i].vz += std::get<2>(forces[i]) / bodies[i].mass * DT;

        // Update positions (Hint: this step involves forces[i], bodies[i].mass, and DT)
        bodies[i].x += bodies[i].vx * DT;
        bodies[i].y += bodies[i].vy * DT;
        bodies[i].z += bodies[i].vz * DT;

        // If a body hits a boundary, reverse it's velocity
        if ((bodies[i].x >= WIDTH) || (bodies[i].x <= 0)) {
            bodies[i].vx = -(bodies[i].vx);
        }
        if ((bodies[i].y >= HEIGHT) || (bodies[i].y <= 0)) {
            bodies[i].vy = -(bodies[i].vy);
        }
        if ((bodies[i].z >= DEPTH) || (bodies[i].z <= 0)) {
            bodies[i].vz = -(bodies[i].vz);
        }
    }
}

void initialize_bodies() {
    // Initialize NUM_BODIES particles
    srand(time(0)); // Ensures different random number on each run

    //#pragma omp parallel for schedule(static) //num_threads(8)
    for (int i = 0; i < NUM_BODIES; i++) {
        bodies[i].x = rand() % int(WIDTH);  // Random num in range 0 - WIDTH
        bodies[i].y = rand() % int(HEIGHT);
        bodies[i].z = rand() % int(DEPTH);

        bodies[i].vx = (rand() % (2 * VELOCITY_RANGE)) - VELOCITY_RANGE;  // Random num in range -VELOCITY_RANGE to VELOCITY_RANGE
        bodies[i].vy = (rand() % (2 * VELOCITY_RANGE)) - VELOCITY_RANGE;
        bodies[i].vz = (rand() % (2 * VELOCITY_RANGE)) - VELOCITY_RANGE;

        bodies[i].mass = (rand() % int(MAX_MASS)) + 1;  // Random num in range 1 - (MAX_MASS + 1)
    }
}

void save_to_csv(std::ofstream &file, int step) {
    for (int i = 0; i < NUM_BODIES; i++) {
        file << step << "," << i << "," << bodies[i].x << "," << bodies[i].y << "," << bodies[i].z << "\n";
    }
}

void run_simulation() {
    std::ofstream file("nbody_output.csv");
    file << "step,id,x,y,z\n";
    
    for (int step = 0; step < NUM_STEPS; step++) {
        compute_forces();
        update_positions();
        save_to_csv(file, step);
    }
    
    file.close();
}

void print_bodies() {
    // For testing. Prints all bodies info
    for (int i = 0; i < NUM_BODIES; i++) {
        std::cout << "Body: " << i << ", ";

        std::cout << (bodies[i].x) << ", ";
        std::cout << (bodies[i].y) << ", ";
        std::cout << (bodies[i].z) << ", ";

        std::cout << (bodies[i].vx) << ", ";
        std::cout << (bodies[i].vy) << ", ";
        std::cout << (bodies[i].vz) << ", ";

        std::cout << bodies[i].mass << "\n";
    }
}

int main(int argc, char *argv[]) {
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
    std::cout << "\nParallel runtime: " << elapsed.count() << " seconds\n";

    std::cout << "Simulation complete. Data saved to nbody_output.csv\n\n";
    return 0;
}
