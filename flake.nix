{
  description = "Machine learning experiments for mathematics";

  inputs = {
    nixpkgs = {
      url = github:NixOS/nixpkgs;
    };
    utils = {
      url = github:numtide/flake-utils;
    };
  };


  outputs = {self, nixpkgs, utils}:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          allowUnfree = true;
          inherit system;  
        };
      in {
        devShells.default = pkgs.mkShell {
          name = "python shell";

          buildInputs = with pkgs; [
            python310
	    python310Packages.matplotlib
            python310Packages.pytorch
          ];


          shellHook = ''
          '';

        };
      }
    );
}
