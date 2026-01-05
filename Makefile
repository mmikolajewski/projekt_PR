NVCC = nvcc
CXX  = g++

CXXFLAGS = -O3
NVCCFLAGS = -O3

OBJS = program_PR.o kernel_a.o kernel_b.o kernel_c.o kernel_d.o

all: program_PR

program_PR.o: projekt_PR.cu
	$(NVCC) $(NVCCFLAGS) -c projekt_PR.cu -o program_PR.o
kernel_a.o: kernel_a.cu
	$(NVCC) $(NVCCFLAGS) -c kernel_a.cu -o kernel_a.o
kernel_b.o: kernel_b.cu
	$(NVCC) $(NVCCFLAGS) -c kernel_b.cu -o kernel_b.o
kernel_c.o: kernel_c.cu
	$(NVCC) $(NVCCFLAGS) -c kernel_c.cu -o kernel_c.o
kernel_d.o: kernel_d.cu
	$(NVCC) $(NVCCFLAGS) -c kernel_d.cu -o kernel_d.o   

program_PR: $(OBJS)
	$(NVCC) $(OBJS) -o projekt_PR

clean:
	rm -f *.o projekt_PR