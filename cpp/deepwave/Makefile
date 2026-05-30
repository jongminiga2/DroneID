# Makefile for remove_turbo — standalone DJI DroneID turbo decoder
#
# Usage:
#   make                          # native (Linux x86-64)
#   make CROSS=arm-linux-gnueabihf-   # ARM 32-bit cross-compile
#   make CROSS=aarch64-linux-gnu-     # ARM 64-bit cross-compile
#   make clean

CROSS   ?=
CXX     := $(CROSS)g++
CXXFLAGS := -O2 -std=c++11 -Wall -Wextra
TARGET  := remove_turbo
SRC     := remove_turbo.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
