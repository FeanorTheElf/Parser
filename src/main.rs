#![feature(fn_traits)]
#![allow(unused)]

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
//use language::scope::{ ScopeTable, fill_sope_info_func };

fn main() {
	let mut stream = lex("fn test(a: int[], b: int[],): int { let length: int = len(a); return a[0] + b[0] + length; }".to_owned());
	let function = FunctionNode::parse(&mut stream);
	println!("{:?}", function);
}