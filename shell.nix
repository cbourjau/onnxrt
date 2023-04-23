# Taken from https://nixos.wiki/wiki/Rust#Installation_via_rustup
# and https://discourse.nixos.org/t/compile-a-rust-binary-on-macos-dbcrossbar/8612
{ pkgs ? import <nixpkgs> {} }:
let
  my-python = pkgs.python3;
  python-with-my-packages = my-python.withPackages (p: with p; [
    # onnxruntime
    pytest
    numpy
    # other python packages you want
  ]);
in
pkgs.mkShell rec {
  buildInputs = with pkgs; [
    clang
    llvmPackages_latest.llvm
    llvmPackages_latest.bintools
    llvmPackages_latest.lld
    darwin.apple_sdk.frameworks.Security
    python3
    python310Packages.flake8
    mypy
    # micromamba
    maturin
    black
    python-with-my-packages
    py-spy
  ] ++ lib.optional stdenv.isDarwin libiconv;

  # https://github.com/rust-lang/rust-bindgen#environment-variables
  LIBCLANG_PATH= pkgs.lib.makeLibraryPath [ pkgs.llvmPackages_latest.libclang.lib ];
  HISTFILE=toString ./.history;
  RUST_BACKTRACE=1;
  VIRTUAL_ENV=toString ./.venv;
  shellHook = ''
    python -m venv $VIRTUAL_ENV
    source $VIRTUAL_ENV/bin/activate
    pip install onnxruntime pytest
  '';
}
