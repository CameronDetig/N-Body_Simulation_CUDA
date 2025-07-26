CXXFLAGS=-std=c++17
LDLIBS=-lpthread

all: nbody_parallel
	./nbody_parallel

nbody_parallel:
	g++-14 -std=c++17 -fopenmp nbody_parallel.cpp -o nbody_parallel

run_nbody_parallel:
	rm nbody_parallel
	g++-14 -std=c++17 -fopenmp nbody_parallel.cpp -o nbody_parallel
	./nbody_parallel

clean:
	rm nbody_parallel