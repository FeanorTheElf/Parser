#![allow(warnings)]
#![feature(fn_traits)]

mod tokens;
mod ast;
mod lexer;
#[macro_use]
mod parser_gen;
mod parser;
mod rust_backend;

use parser_gen::Parse;
use lexer::lex;
use tokens::*;
use ast::*;

fn main() {
	let mut ts = lex("
		fn foo(a: int[], ): int[] {
			let b: int[] = new int[len(a)];
			return b;
		}".to_string());
	let tt = Function::parse(&mut ts);
	println!("{:?}", tt);
}