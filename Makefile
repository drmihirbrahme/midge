# midge engine build
CC      ?= gcc
# portable baseline on x86; use ARCH=native to tune, ARCH= to disable
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  ARCH ?= x86-64-v2
else
  ARCH ?=
endif
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-function
LDFLAGS ?= -lm
# OpenMP if the compiler supports it (Apple clang doesn't by default:
# the engine then runs single-threaded — use ARCH/OMP overrides or midge-mlx)
OMP_OK := $(shell echo 'int main(){return 0;}' | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo yes)
ifeq ($(OMP_OK),yes)
  CFLAGS  += -fopenmp
  LDFLAGS += -fopenmp
else
  CFLAGS  += -Wno-unknown-pragmas
endif

ifeq ($(ARCH),native)
  CFLAGS += -march=native
else ifneq ($(ARCH),)
  CFLAGS += -march=$(ARCH)
endif

all: midged

midged: engine/midge.c engine/mjson.h engine/mten.h engine/mkern.h
	$(CC) $(CFLAGS) engine/midge.c -o midged $(LDFLAGS)

test: midged
	python3 tools/validate.py

clean:
	rm -f midged

.PHONY: all test test-mlx test-server test-ui clean

test-mlx: midged
	python3 tools/validate_mlx.py

test-server: midged
	python3 tools/validate_server.py

test-ui:
	python3 tools/validate_ui.py
