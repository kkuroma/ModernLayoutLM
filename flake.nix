{
  description = "modern-layoutlm";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/9bc02893134c733dd85de46ee4fb2fac696b5529";
  outputs = { self, nixpkgs }:
  let
    pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
    lib = nixpkgs.lib;
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      packages = with pkgs; [
        python3
        uv
        ruff
        black
        pyright
        cudaPackages.cuda_cudart
        cudaPackages.libcublas
        cudaPackages.cuda_cccl
        cudaPackages.cuda_nvcc
        # Merged nvcc + crt + cudart + cublas, matching torch's cu130 ABI. flash-attn is built
        # from source because upstream ships no cp314 wheel, and the SPLIT cuda packages are not
        # enough: cuda_nvcc alone lacks crt/host_config.h, which lives in cuda_crt. cudatoolkit
        # is the symlinkJoin of them and is the only valid CUDA_HOME.
        cudaPackages_13_0.cudatoolkit
        git
      ];
      DEV_SHELL = "modern-layoutlm";
      LD_LIBRARY_PATH = "/run/opengl-driver/lib:" + lib.makeLibraryPath (with pkgs; [ stdenv.cc.cc.lib zlib libGL cudaPackages.cuda_cudart cudaPackages.libcublas cudaPackages.cuda_cccl ]);
      # torch 2.13 routes ModernBERT's RoPE through a triton kernel, so triton has to work or
      # every CUDA training run dies on its first step. Two things break it on NixOS.
      #
      # Triton locates libcuda by shelling out to /sbin/ldconfig, which NixOS does not have. It
      # checks this variable first, so setting it skips that call entirely.
      TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
      # Triton also ships its own ptxas, built against a generic linux interpreter that NixOS
      # cannot execute. Point it at the nix one instead.
      TRITON_PTXAS_PATH = "${pkgs.cudaPackages.cuda_nvcc}/bin/ptxas";
      # For building flash-attn. Not a runtime dependency: once built, uv caches the wheel.
      #   uv pip install --no-build-isolation flash-attn
      # It stays out of pyproject.toml deliberately. The build takes ~110 minutes, and on a
      # machine without this toolchain it would fail the whole `uv sync`, which is the same way
      # paddlepaddle made the lockfile unresolvable.
      CUDA_HOME = "${pkgs.cudaPackages_13_0.cudatoolkit}";
      TORCH_CUDA_ARCH_LIST = "8.6";
      # cicc peaks near 2.7GB per job, so 12 fits comfortably in 62GB. 8 underused the 16 cores.
      MAX_JOBS = "12";
      shellHook = ''
        if [ -d .venv ]; then
          export VIRTUAL_ENV="$PWD/.venv"
          export PATH="$PWD/.venv/bin:$PATH"
        fi
        [[ $- == *i* ]] && exec zsh
      '';
    };
  };
}
