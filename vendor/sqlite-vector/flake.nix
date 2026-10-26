{
  description = "SQLite Vector Extension";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        stdenv = pkgs.stdenv;
        version = "0.9.53";
      in
      {
        packages.default = stdenv.mkDerivation {
          pname = "sqlite-vector";
          inherit version;

          src = pkgs.lib.cleanSource ./.;

          makeFlags = [
            "CC=${stdenv.cc.targetPrefix}cc"
          ] ++ pkgs.lib.optionals stdenv.isDarwin [
            "ARCH=${if stdenv.hostPlatform.isAarch64 then "arm64" else "x86_64"}"
          ];

          installPhase = "install -D dist/vector* -t $out/lib";

          checkInputs = [ pkgs.sqlite ];
          doCheck = true;
          checkPhase = ''
            make test
          '';
        };

        devShells.default =
          let
            sqlite-vector = self.packages.${system}.default;
          in
          pkgs.mkShell {
            packages = [
              pkgs.sqlite
              pkgs.gnumake
              sqlite-vector
            ];

            shellHook = ''
              export SQLITE_VECTOR_LIB="${sqlite-vector}/lib/vector${stdenv.hostPlatform.extensions.sharedLibrary}"
              echo "SQLite Vector extension available at: $SQLITE_VECTOR_LIB"
              echo "Load it in sqlite3 with: .load $SQLITE_VECTOR_LIB"
            '';
          };
      }
    );
}
