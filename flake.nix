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
        git
      ];
      DEV_SHELL = "modern-layoutlm";
      LD_LIBRARY_PATH = "/run/opengl-driver/lib:" + lib.makeLibraryPath (with pkgs; [ stdenv.cc.cc.lib zlib libGL cudaPackages.cuda_cudart cudaPackages.libcublas cudaPackages.cuda_cccl ]);
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
