#![feature(fn_traits)]
#![allow(unused)]
#![feature(test)]

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

use lexer::tokens::*;
use parser::Parse;
use lexer::lexer::lex;
use parser::ast::*;

fn main() {
	let mut stream = lex("fn len(a: int[],): int native;");
	let function = FunctionNode::parse(&mut stream).unwrap();
}
