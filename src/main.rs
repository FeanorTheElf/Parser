#![feature(test)]
#![feature(trait_alias)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]

extern crate itertools;
extern crate take_mut;

#[cfg(test)]
extern crate test;

#[macro_use]
mod util;
#[macro_use]
mod language;

mod lexer;
mod parser;

mod analysis;
mod transformation;

mod cuda;

use language::prelude::*;
use lexer::lexer::lex;
use parser::Parser;
use language::backend::Backend;

fn main() {
    let mut program = Program::parse(&mut lex("
		fn main() {
			
		}
	"))
	.unwrap();
	let mut out = "".to_owned();
    let mut target = language::backend::StringWriter::new(&mut out);
	let mut writer = language::backend::CodeWriter::new(&mut target);
	let mut cuda_backend = cuda::backend::CudaBackend{};
	cuda_backend.init().unwrap();
	cuda_backend.transform_program(&mut program).unwrap();
	cuda_backend.generate(&program, &mut writer).unwrap();
	println!("{}", out);
}
