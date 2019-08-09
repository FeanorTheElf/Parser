#![allow(warnings)]
#![feature(fn_traits)]

#[macro_use]
mod util;
mod lexer;
mod parser;
mod language;
mod la;

use lexer::tokens::*;
use parser::parser_gen::Parse;
use lexer::lexer::lex;
use parser::ast::*;
use la::simplex::experiment;

fn main() {
	Function::parse(&mut lex("fn test(a: int): int { return a; }".to_owned())).unwrap();
}