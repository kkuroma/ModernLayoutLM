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
      shellHook = ''
        # Both HF caches belong on the vault. Downloads pass an explicit cache_dir, but the hub
        # weights and the arrow conversions of data/processed do not, and those defaults quietly
        # put 325GB in ~/.cache/huggingface on the SSD before anyone noticed.
        export HF_HOME="$PWD/data/raw/hf-cache"
        if [ -d .venv ]; then
          export VIRTUAL_ENV="$PWD/.venv"
          export PATH="$PWD/.venv/bin:$PATH"
        fi
        [[ $- == *i* ]] && exec zsh
      '';
    };
  };
}
