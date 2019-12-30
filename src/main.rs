#![feature(fn_traits)]
#![feature(test)]
#![allow(dead_code)]

extern crate take_mut;

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

fn main() {
	let mut stream = lex("fn len(a: int[],): int native;");
	let function = FunctionNode::parse(&mut stream).unwrap();
	println!("{:?}", function)
}
