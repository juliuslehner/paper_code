# Makefile for compiling FFTW precision and performance testing programs

# Compiler
CXX = g++

# Source files
SRC_PRECISION = fftw_precision_testing_1d.cpp
SRC_PERFORMANCE = fftw_performance_testing_1d.cpp

# 2D Source files
SRC_PRECISION_2D = fftw_precision_testing_2d.cpp
SRC_PERFORMANCE_2D = fftw_performance_testing_2d.cpp

# Libraries
LDFLAGS = -Wl,-rpath=/papi/lib
INCLUDES = -I/papi/include
LIBDIRS = -L/papi/lib
LIBS = -lfftw3_threads -lfftw3f_threads -lfftw3l_threads -lfftw3f -lfftw3 -lfftw3l -lm -fopenmp -lpapi

# Default target
all: precision_double_1d performance_double_1d precision_double_2d performance_double_2d

# Check if MULTI_THREAD flag is set
ifdef mt
CXXFLAGS += -DMULTI_THREADS
endif

# Single precision builds
precision_single_1d: CXXFLAGS += -DSINGLE_PRECISION
precision_single_1d: fftw_precision_single_1d

performance_single_1d: CXXFLAGS += -DSINGLE_PRECISION
performance_single_1d: fftw_performance_single_1d

precision_single_2d: CXXFLAGS += -DSINGLE_PRECISION
precision_single_2d: fftw_precision_single_2d

performance_single_2d: CXXFLAGS += -DSINGLE_PRECISION
performance_single_2d: fftw_performance_single_2d

# Double precision builds
precision_double_1d: CXXFLAGS += -DDOUBLE_PRECISION
precision_double_1d: fftw_precision_double_1d

performance_double_1d: CXXFLAGS += -DDOUBLE_PRECISION
performance_double_1d: fftw_performance_double_1d

precision_double_2d: CXXFLAGS += -DDOUBLE_PRECISION
precision_double_2d: fftw_precision_double_2d

performance_double_2d: CXXFLAGS += -DDOUBLE_PRECISION
performance_double_2d: fftw_performance_double_2d

# Long double precision builds (only for performance testing)
performance_longdouble_1d: CXXFLAGS += -DLONGDOUBLE_PRECISION
performance_longdouble_1d: fftw_performance_longdouble_1d

performance_longdouble_2d: CXXFLAGS += -DLONGDOUBLE_PRECISION
performance_longdouble_2d: fftw_performance_longdouble_2d

# Build rules for precision testing
fftw_precision_single_1d: $(SRC_PRECISION)
	$(CXX) -o fftw_precision_single_1d $(SRC_PRECISION) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS)

fftw_precision_double_1d: $(SRC_PRECISION)
	$(CXX) -o fftw_precision_double_1d $(SRC_PRECISION) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS)

fftw_precision_single_2d: $(SRC_PRECISION_2D)
	$(CXX) -o fftw_precision_single_2d $(SRC_PRECISION_2D) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS)

fftw_precision_double_2d: $(SRC_PRECISION_2D)
	$(CXX) -o fftw_precision_double_2d $(SRC_PRECISION_2D) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS)

# Build rules for performance testing
fftw_performance_single_1d: $(SRC_PERFORMANCE)
	$(CXX) -o fftw_performance $(SRC_PERFORMANCE) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) $(LDFLAGS)

fftw_performance_double_1d: $(SRC_PERFORMANCE)
	$(CXX) -o fftw_performance $(SRC_PERFORMANCE) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) $(LDFLAGS)

fftw_performance_longdouble_1d: $(SRC_PERFORMANCE)
	$(CXX) -o fftw_performance $(SRC_PERFORMANCE) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) $(LDFLAGS)

fftw_performance_single_2d: $(SRC_PERFORMANCE_2D)
	$(CXX) -o fftw_performance $(SRC_PERFORMANCE_2D) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) $(LDFLAGS)

fftw_performance_double_2d: $(SRC_PERFORMANCE_2D)
	$(CXX) -o fftw_performance $(SRC_PERFORMANCE_2D) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) $(LDFLAGS)

fftw_performance_longdouble_2d: $(SRC_PERFORMANCE_2D)
	$(CXX) -o fftw_performance $(SRC_PERFORMANCE_2D) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) $(LDFLAGS)

# Clean up
clean:
	rm -f fftw_precision_single_1d fftw_precision_double_1d fftw_precision_single_2d fftw_precision_double_2d \
	       fftw_performance_single_1d fftw_performance_double_1d fftw_performance_longdouble_1d \
           fftw_performance_single_2d fftw_performance_double_2d fftw_performance_longdouble_2d
