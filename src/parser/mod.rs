use super::lexer::tokens::Stream;
use super::language::error::CompileError;
use super::language::position::TextPosition;

pub trait Parseable
{
	type ParseOutputType;
}

pub trait Build<T>: Parseable
{
	fn build(pos: TextPosition, params: T) -> Self::ParseOutputType;
}

pub trait Parser: Parseable
{
	fn is_applicable(stream: &Stream) -> bool;
	fn parse(stream: &mut Stream) -> Result<Self::ParseOutputType, CompileError>;
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
