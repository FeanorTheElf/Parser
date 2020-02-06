use super::lexer::tokens::Stream;
use super::language::error::CompileError;

pub trait Parse 
{
	fn guess_can_parse(stream: &Stream) -> bool;
	fn parse(stream: &mut Stream) -> Result<Box<Self>, CompileError>;
}

pub trait Build<T>
{
	fn build(params: T) -> Self;
}

#[macro_use]
pub mod parser_gen;
// pub mod obj_type;
// pub mod print;
// pub mod visitor;
// #[macro_use]
// pub mod ast;
// pub mod ast_expr;
// pub mod ast_func;
// pub mod parser;
// pub mod prelude;
