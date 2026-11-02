{
  description = "Valkey-backed L2 for an LLM block KV cache";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system} system);
      py = pkgs: pkgs.python3.withPackages (ps: [ ps.numpy ]);
    in {
      devShells = forAll (pkgs: _: {
        default = pkgs.mkShell {
          packages = [ (py pkgs) pkgs.valkey ];
          shellHook = ''
            echo "start a store: valkey-server --port 6390 --save '' --appendonly no &"
            echo "then:         python3 tests/test_tiered.py"
          '';
        };
      });

      checks = forAll (pkgs: _: {
        tests = pkgs.runCommand "valkey-kv-l2-tests" {
          buildInputs = [ (py pkgs) pkgs.valkey ];
        } ''
          cp -r ${self} work && chmod -R u+w work && cd work
          valkey-server --port 6390 --save "" --appendonly no --daemonize yes
          sleep 2
          python3 tests/test_tiered.py
          valkey-cli -p 6390 shutdown nosave || true
          touch $out
        '';
      });

      packages = forAll (pkgs: _: {
        default = pkgs.runCommand "valkey-kv-l2" { } ''
          cp -r ${self} $out
        '';
      });
    };
}
