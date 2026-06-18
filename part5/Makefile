CUDA_ARCH := sm_86
NVCC := nvcc
TPB_LS ?= 256
TPB_EQ ?= 256
DEFAULT_PATTERNS ?= 16

NVCCFLAGS := -O2 -std=c++17 -arch=$(CUDA_ARCH) -lineinfo -Xptxas -v \
	-DPART5_TPB_LS=$(TPB_LS) -DPART5_TPB_EQ=$(TPB_EQ)
HOST_CXX := g++
HOST_CXXFLAGS := -O2 -std=c++17

SRC := main.cu
CPU_SRC := main_cpu.cpp
GEN_SRC := generate_ofdm_multi.cpp
BIN := main
CPU_BIN := main_cpu
GEN_BIN := generate_ofdm_multi
PTX := main.ptx
DATA_FILE := ../data/ofdm_input_multi.bin

all: $(BIN) $(CPU_BIN)

$(BIN): $(SRC) $(DATA_FILE)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(CPU_BIN): $(CPU_SRC) $(DATA_FILE)
	$(HOST_CXX) $(HOST_CXXFLAGS) $< -o $@

$(GEN_BIN): $(GEN_SRC)
	$(HOST_CXX) $(HOST_CXXFLAGS) $< -o $@

$(DATA_FILE): $(GEN_BIN)
	mkdir -p ../data
	./$(GEN_BIN) $(DATA_FILE) $(DEFAULT_PATTERNS)

ptx: $(SRC)
	$(NVCC) --ptx -O2 -std=c++17 -arch=$(CUDA_ARCH) $< -o $(PTX)

run: $(BIN)
	./$(BIN) $(DATA_FILE)

run_cpu: $(CPU_BIN)
	./$(CPU_BIN) $(DATA_FILE)

profile: $(BIN)
	ncu --set basic ./$(BIN) $(DATA_FILE) 1 1

sweep: $(BIN) $(GEN_BIN)
	./run_experiments.sh

clean:
	rm -f $(BIN) $(CPU_BIN) $(GEN_BIN) $(PTX)

.PHONY: all ptx run run_cpu profile sweep clean
