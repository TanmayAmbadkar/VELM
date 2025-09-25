CXX = g++
# First, add the conda environment's include directory to your CFLAGS.
# You can find the path by running `conda run --prefix <your_env> echo $CONDA_PREFIX/include`
# The exact path will depend on your environment and system.
CONDA_INCLUDE = /scratch1/tsa5252/anaconda3/envs/velmenv/include
CFLAGS = -I . -I $(HOME) -I $(CONDA_INCLUDE) -g -O3 -std=c++11

POLAR_HOME = ../../POLAR
FLOWSTAR_HOME = ../../flowstar/flowstar-toolbox

# The LIBS variable is already correct for linking.
LIBS = -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lpthread

# Add the conda environment's lib directory to your LINK_FLAGS
# so the linker can find the correct version of libmpfr.so.
CONDA_LIB = /scratch1/tsa5252/anaconda3/envs/velmenv/lib
LINK_FLAGS = -g -L$(POLAR_HOME) -L$(FLOWSTAR_HOME) -L/usr/local/lib -L$(CONDA_LIB)
CURRENT_DIR := $(shell pwd)

all: pendulum

pendulum: pendulum.o
	@echo "Current directory is: $(CURRENT_DIR)"
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)
	cp pendulum ../../../results/programs/

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o pendulum
