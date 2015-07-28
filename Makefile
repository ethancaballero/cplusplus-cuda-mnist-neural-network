#Maakefile for Macs, will need to modify for the Network to run on linux



#NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch=sm_20
#LD_FLAGS    = -lcudart -L/usr/local/cuda/lib

NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib

OBJS = Network.o Node.o main.o 
EXENAME = Network


COMPILER = nvcc
COMPILER_OPTS = -c
LINKER = nvcc




default : $(EXENAME)

Node.o : Network.h Node.h Node.cu
	$(COMPILER) -c -o $@ Node.cu $(NVCC_FLAGS)

Network.o : Network.cu Network.h Node.h
	$(COMPILER) -c -o $@ Network.cu $(NVCC_FLAGS)

main.o : main.cu Network.h Node.h 
	$(COMPILER) -c -o $@ main.cu $(NVCC_FLAGS)
	
$(EXENAME) : $(OBJS)
	$(LINKER) $(OBJS) -o $(EXENAME) $(LD_FLAGS)
	
clean:
	-rm -f *.o $(EXENAME)