#![feature(fn_traits)]
#![feature(specialization)]
#![feature(never_type)]
#![feature(test)]
#![allow(unused)]

#[cfg(test)]
extern crate test;

#[macro_use]
mod util;
mod lexer;
mod parser;
mod transformer;
mod language;
mod la;
mod backend;

use lexer::tokens::*;
use parser::Parse;
use lexer::lexer::lex;
use parser::ast::*;
use language::scope::{ ScopeTable, annotate_sope_info_func };
use language::symbol::{ SymbolTable, annotate_symbols_function };

fn main() {
	let mut stream = lex("fn len(a: int[],): int native;");
	let function = FunctionNode::parse(&mut stream).unwrap();
	let mut scopes = ScopeTable::new();
	annotate_sope_info_func(&function, &mut scopes).unwrap();
	let mut symbols = SymbolTable::new();
	annotate_symbols_function(&function, &scopes, &mut symbols).unwrap();
	println!("{:?}", symbols.get_type(&function.ident));
}
