include ./make.in

EXEC = pme

SRC = $(wildcard *.c) 
OBJS := $(addsuffix .o, $(basename $(SRC)))

CFLAGS = -g -openmp
CFLAGS += -debug parallel -Wall -w2 
CFLAGS += -restrict
CFLAGS += -Wsign-compare -Wunknown-pragmas -Wreturn-type
CFLAGS += -Wunused-variable -Wuninitialized -Wmissing-prototypes -Wmissing-declarations

NVCC = nvcc -m64 -arch=sm_35 -G -Xcompiler "-g -pg -O2" 
CUFLAGS =
CUSRC = $(wildcard *.cu)
CUOBJS := $(addsuffix .o, $(basename $(CUSRC)))
CUOBJS += dlink.o

INC = -I.
CUINC = -I.

## blas library ##      
INC += -I${BLAS_INCDIR} -I${CUDA_INCDIR}
LDFLAGS += -L${BLAS_LIBDIR} -L${OMP_LIBDIR}
LIBS += ${BLAS_LIBS}

## cuda library ##
CUINC += -I${CUDA_INCDIR} -I${BLAS_INCDIR}
CUFLAGS += -L${CUDA_LIBDIR}
CULIBS += ${CUDA_LIBS}

## Defines ##
DEFS = -DGPU_TRACE

all: ${EXEC}

# $(EXEC): $(OBJS) Makefile
#	$(CC) ${CFLAGS} ${LDFLAGS} $(OBJS) -o $(EXEC) ${LIBS} ${DEFS}

$(EXEC): $(OBJS) $(CUOBJS) Makefile
	$(CC) ${CFLAGS} ${LDFLAGS} $(OBJS) $(CUOBJS) -o $(EXEC) ${LIBS} ${DEFS} ${CULIBS} ${CUFLAGS}

%.o : %.c Makefile
	$(CC) ${CFLAGS} ${INC} -c $< -o $@ ${DEFS}

%.o : %.cu Makefile
	$(NVCC) -G -dc ${CUFLAGS} ${CUINC} ${CULIBS} -c $< -o $@ ${DEFS}
	$(NVCC) -dlink $@ -o dlink.o -lcudadevrt

clean:
	rm -f *.o *.d *~ ${EXEC}
