#![allow(warnings)]
#![feature(fn_traits)]

#[macro_use]
mod parser;
mod language;
mod simplex;

use parser::tokens::*;
use parser::parser_gen::Parse;
use language::lexer::lex;
use language::ast::*;
use simplex::experiment;

fn main() {
	experiment();
}