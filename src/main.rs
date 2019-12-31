#![feature(fn_traits)]
#![feature(test)]
#![allow(dead_code)]

extern crate take_mut;
extern crate itertools;

#[cfg(test)]
extern crate test;

#[macro_use]
mod util;
mod lexer;
mod parser;
mod language;
mod la;
mod backend;

use parser::Parse;
use lexer::lexer::lex;
use parser::prelude::*;
use language::inline::inline;

fn main() {
	let mut program = Program::parse(&mut lex("
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
	")).unwrap();
	inline(&mut *program);
	println!("{}", program);
}
