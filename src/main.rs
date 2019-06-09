#![allow(warnings)]
#![feature(fn_traits)]

#[macro_use]
mod util;
#[macro_use]
mod parser;
mod language;
mod la;

use parser::tokens::*;
use parser::parser_gen::Parse;
use language::lexer::lex;
use language::ast::*;
use la::simplex::experiment;

fn main() {
	experiment();
}