![Rust](https://img.shields.io/badge/-Rust-red?logo=rust&logoColor=white&style=plastic)
![TOML](https://img.shields.io/badge/-Toml-blue?logo=toml&style=plastic)

AVX-512 Benchmark Comparison written in RUST

IMPORTANT : Make sure your CPU has AVX-512 feature


## Requirement
This SIMD features still experimental, so you need to use Rust nightly

``` 
rustup toolchain install nightly 

rustup default nightly

```

if Rust Analyzer still giving error messages, run this command on project folder:
` rustup override set nightly `

Close folder on IDE, reopen folder


## Benchmark

` cargo bench `
