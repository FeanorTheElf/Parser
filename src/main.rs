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
use language::scope::{ ScopeTable, fill_sope_info_func };

fn main() {
	la::diophantine::experiment();
}