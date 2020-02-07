use super::lexer::tokens::Stream;
use super::language::error::CompileError;
use super::language::position::TextPosition;

pub trait Buildable
{
    type OutputType;
}

pub trait Build<T>: Buildable
{
	fn build(pos: TextPosition, params: T) -> Self::OutputType;
}

pub trait Parse: Buildable
{
	fn guess_can_parse(stream: &Stream) -> bool;
	fn parse(stream: &mut Stream) -> Result<Self::OutputType, CompileError>;
}

#[macro_use]
pub mod parser_gen;
pub mod parser;
// pub mod obj_type;
// pub mod print;
// pub mod visitor;
// #[macro_use]
// pub mod ast;
// pub mod ast_expr;
// pub mod ast_func;
// pub mod parser;
// pub mod prelude;
