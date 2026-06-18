CUDA_ARCH := sm_86
NVCC := nvcc
TPB_LS ?= 256
TPB_EQ ?= 256
NVCCFLAGS := -O2 -std=c++17 -arch=$(CUDA_ARCH) -lineinfo -Xptxas -v \
	-DPART4_TPB_LS=$(TPB_LS) -DPART4_TPB_EQ=$(TPB_EQ)

HOST_CXX := g++
HOST_CXXFLAGS := -O2 -std=c++11
SRC := main.cu
BIN := main
PTX := main.ptx
GEN_SRC := ../data_gen/generate_ofdm_data.cpp
GEN_BIN := ../data_gen/generate_ofdm_data
DATA_FILE := ../data/ofdm_input.bin

all: $(BIN)

$(BIN): $(SRC) $(DATA_FILE)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(GEN_BIN): $(GEN_SRC)
	$(HOST_CXX) $(HOST_CXXFLAGS) $< -o $@

$(DATA_FILE): $(GEN_BIN)
	mkdir -p ../data
	$(GEN_BIN) $(DATA_FILE)

ptx: $(SRC)
	$(NVCC) --ptx -O2 -std=c++17 -arch=$(CUDA_ARCH) $< -o $(PTX)

run: $(BIN)
	./$(BIN)

profile: $(BIN)
	ncu --set basic ./$(BIN) ../data/ofdm_input.bin 1 1

sweep:
	./run_experiments.sh

clean:
	rm -f $(BIN) $(PTX) $(GEN_BIN)

.PHONY: all ptx run profile sweep clean
