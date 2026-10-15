{
  description = "LayNii development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      devShells = forAllSystems (pkgs: {
        # gccStdenv makes `g++` (hardcoded in the Makefile) real GCC on macOS
        # too, and gives it the include/lib flags for buildInputs.
        default = pkgs.mkShell.override { stdenv = pkgs.gccStdenv; } {
          buildInputs = [ pkgs.zlib ];
          packages = [
            pkgs.gnumake
            pkgs.codespell
            pkgs.gh
          ];
        };
      });
    };
}
