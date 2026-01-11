NVCC = nvcc
NVCCFLAGS = -O3 -Icuda_pomoc

# Nazwy programów
TARGET1 = nwys
TARGET2 = paramk
TARGET3 = metryka
TARGET4 = projekt_PR

# Pliki główne
SRC1 = nwys.cu
SRC2 = paramk.cu
SRC3 = metryka.cu
SRC4 = projekt_PR.cu

# Pliki wspólne
COMMON = kernel_a.cu kernel_b.cu kernel_c.cu kernel_d.cu

# Obiekty
OBJ1 = $(SRC1:.cu=.o)
OBJ2 = $(SRC2:.cu=.o)
OBJ3 = $(SRC3:.cu=.o)
OBJ4 = $(SRC4:.cu=.o)
OBJ_COMMON = $(COMMON:.cu=.o)

# Domyślny cel
all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)
# Linkowanie
$(TARGET1): $(OBJ1) $(OBJ_COMMON)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(TARGET2): $(OBJ2) $(OBJ_COMMON)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(TARGET3): $(OBJ3) $(OBJ_COMMON)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(TARGET4): $(OBJ4) $(OBJ_COMMON)
	$(NVCC) $(NVCCFLAGS) -o $@ $^
# Kompilacja .cu -> .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Sprzątanie
clean:
	rm -f *.o $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)