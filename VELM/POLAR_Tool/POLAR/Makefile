CXX = g++

# Adjust paths to use the Conda environment's include and lib directories
# This ensures that headers like <mpfr.h> and <gmp.h> are found correctly.
# The user must still ensure that makefile.local points correctly to flowstar.
HOME = /usr/local/include
LIB_HOME = ../flowstar/flowstar-toolbox
CONDA_INCLUDE = $(CONDA_PREFIX)/include
CONDA_LIB = $(CONDA_PREFIX)/lib

# Define the libraries to link against.
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lpthread

# Add Conda's include directory to CFLAGS.
# The -I flag tells the compiler where to find header files.
CFLAGS = -I . -I $(HOME) -I$(CONDA_INCLUDE) -g -O3 -std=c++11

# Add Conda's lib directory to LINK_FLAGS.
# The -L flag tells the linker where to find shared or static library files.
LINK_FLAGS = -g -L$(LIB_HOME) -L/usr/local/lib -L$(CONDA_LIB)

OBJS = NeuralNetwork.o BernsteinPoly.o Neuron.o PolarSetting.o System.o Specification.o Polar.o

# --- Targets ---

all: polar_lib polar_tool

# Target to build the executable 'polar_tool'.
# It depends on Polar_tool.o and the other object files.
polar_tool: Polar_tool.o $(OBJS)
	g++ -O3 -w $(LINK_FLAGS) -L./ -o $@ $^ $(LIBS)

# Target to build the static library 'libpolar.a'.
# The `ar` command creates the library from the specified object files.
polar_lib: $(OBJS)
	ar rcs libpolar.a $^

# --- Compilation Rules ---

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<

%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<

%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<

# --- Clean Target ---

clean:
	rm -f *.o libpolar.a polar_tool
