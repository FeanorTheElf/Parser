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
use language::scope::{ ScopeTable, annotate_sope_info_func };
use language::symbol::{ SymbolTable, annotate_symbols_function };

fn main() {
	let mut stream = lex("fn len(a: int[],): int native;".to_owned());
	let function = FunctionNode::parse(&mut stream).unwrap();
	let mut scopes = ScopeTable::new();
	annotate_sope_info_func(&function, &mut scopes).unwrap();
	let mut symbols = SymbolTable::new();
	annotate_symbols_function(&function, &scopes, &mut symbols).unwrap();
	println!("{:?}", symbols.get(&function.ident).symbol_type);
}