CXXFLAGS=-std=c++17
LDLIBS=-lpthread

all: nbody_sequential
	./nbody_sequential

nbody_sequential:
	g++ -std=c++17 nbody_sequential.cpp -o nbody_sequential

run_nbody_sequential:
	rm nbody_sequential
	g++-14 -std=c++17 -fopenmp nbody_sequential.cpp -o nbody_sequential
	./nbody_sequential

clean:
	rm nbody_sequential