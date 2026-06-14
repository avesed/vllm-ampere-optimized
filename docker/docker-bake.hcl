# Local/dev buildx-bake entrypoint for a manual Ampere build (single target), forked
# from upstream docker/docker-bake.hcl and collapsed to one target with the Ampere arch
# list baked in. CI uses scripts/build_image_ampere.sh; this is for one-off local builds.
#
#   git clone --branch v0.23.0 https://github.com/vllm-project/vllm.git vllm
#   python patches/regenerate.py vllm            # apply our patches to the tree
#   VLLM_TAG=v0.23.0 OWNER=<you> docker buildx bake -f docker/docker-bake.hcl openai

variable "VLLM_TAG"             { default = "dev" }
variable "CUDA_VERSION"         { default = "13.0.2" }    # cu130 (driver >= 580.65.06); 12.9.1 for cu129
variable "TORCH_CUDA_ARCH_LIST" { default = "8.0 8.6" }   # all Ampere: A100 sm_80 + 3090/A40/A6000 sm_86
variable "OWNER"                { default = "OWNER" }

target "openai" {
  context    = "vllm"
  dockerfile = "vllm/docker/Dockerfile"
  target     = "vllm-openai"
  platforms  = ["linux/amd64"]
  args = {
    CUDA_VERSION         = CUDA_VERSION
    torch_cuda_arch_list = TORCH_CUDA_ARCH_LIST
    max_jobs             = "8"
    nvcc_threads         = "4"
    RUN_WHEEL_CHECK      = "false"
  }
  tags = ["ghcr.io/${OWNER}/vllm-ampere-optimized:${VLLM_TAG}-ampere"]
  labels = {
    "org.opencontainers.image.source"  = "https://github.com/${OWNER}/vllm-ampere-optimized"
    "org.opencontainers.image.version" = "${VLLM_TAG}-ampere"
  }
  output = ["type=registry"]
}
