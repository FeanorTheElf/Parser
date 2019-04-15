#![allow(warnings)]

mod tokens;
mod ast;
mod lexer;
#[macro_use]
mod parser_gen;
mod parser;

use parser_gen::Parse;
use lexer::lex;
use tokens::*;
use ast::*;

fn main() {

}