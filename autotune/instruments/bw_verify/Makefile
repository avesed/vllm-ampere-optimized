# bw_verify — no-root GDDR bandwidth + integrity probe.
# nvcc may not be on PATH even when the CUDA toolkit is installed; fall back to /usr/local/cuda.
NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
# sm_86 = GA10x (3090/3080/A6000/A5000). Add sm_80 (A100) if needed.
ARCH ?= sm_86

bw_verify: bw_verify.cu
	$(NVCC) -O3 -arch=$(ARCH) $< -o $@

clean:
	rm -f bw_verify

.PHONY: clean
