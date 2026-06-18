GEM5_DIR := /root/gem5
GEM5 := $(GEM5_DIR)/build/RISCV/gem5.opt
CONFIG := $(GEM5_DIR)/configs/deprecated/example/se.py

CC := riscv64-linux-gnu-g++
CFLAGS := -static -O2 -std=c++11
HOST_CXX := g++
HOST_CXXFLAGS := -O2 -std=c++11

SRC := main.cpp
BIN := main
HOST_BIN := main_host
GEN_SRC := ../data_gen/generate_ofdm_data.cpp
GEN_BIN := ../data_gen/generate_ofdm_data
DATA_FILE := ../data/ofdm_input.bin

all: run

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $< -o $@

$(GEN_BIN): $(GEN_SRC)
	$(HOST_CXX) $(HOST_CXXFLAGS) $< -o $@

# Input generation is done on the host before gem5 simulation.
# It prepares a shared binary input file and is not part of the
# simulated workload measured in m5out/stats.txt.
$(DATA_FILE): $(GEN_BIN)
	mkdir -p ../data
	$(GEN_BIN) $(DATA_FILE)

run: $(DATA_FILE) $(BIN)
	$(GEM5) $(CONFIG) -c $(CURDIR)/$(BIN) --cpu-type=TimingSimpleCPU --mem-size=256MB --caches --l1i_size=32kB --l1i_assoc=8 --l1d_size=32kB --l1d_assoc=8 --cacheline=32
	grep -E "simSeconds|simTicks|numCycles|simInsts|simOps|hostSeconds|overallMisses|overallMissRate|system.cpu.cpi|system.cpu.ipc" m5out/stats.txt

host: $(DATA_FILE)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(SRC) -o $(HOST_BIN)
	./$(HOST_BIN)

clean:
	rm -f $(BIN) $(HOST_BIN) $(GEN_BIN)
	rm -rf m5out

.PHONY: all run host clean
