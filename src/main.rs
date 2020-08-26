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

fn main() {
    let program = Program::parse(&mut lex("
		fn min(a: int, b: int,): int {
			if a < b {
				return a;
			}
			return b;
		}
	
		fn max(a: int, b: int,): int {
			if b < a {
				return a;
			}
			return b;
		}

		fn clamp(a: int, lower: int, upper: int,): int {
			return min(max(a, lower, ), upper, );
		}
	"))
    .unwrap();
    println!("{:?}", program);
}
