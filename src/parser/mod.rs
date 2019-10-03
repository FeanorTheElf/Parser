use super::lexer::tokens::Stream;
use super::lexer::error::CompileError;

pub trait Parse {
	fn guess_can_parse(stream: &Stream) -> bool;
	fn parse(stream: &mut Stream) -> Result<Box<Self>, CompileError>;
}

#[macro_use]
pub mod parser_gen;
pub mod ast;
pub mod ast_visitor;
pub mod parser;
pub mod prelude;
