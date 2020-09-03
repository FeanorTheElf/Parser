#![feature(test)]
#![feature(trait_alias)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(iter_map_while)]

//extern crate itertools;
extern crate take_mut;
#[macro_use] 
extern crate serde;
extern crate serde_json;
#[cfg(test)]
extern crate test;

#[macro_use]

mod util;

#[macro_use]

mod language;

mod analysis;
mod lexer;
mod parser;
mod transformation;
mod cuda;
mod cli;

use cli::cli_backend::*;
use cli::cuda::*;

fn main() {
    create_cuda_exe_backend().run(MultiStageBackendOptions::from("E:\\Users\\Simon\\Documents\\Projekte\\Parser\\example")).unwrap();
}
