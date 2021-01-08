use super::language::error::CompileError;
use super::language::position::TextPosition;
use super::lexer::tokens::Stream;

use super::language::types::TypeVec;

pub trait Parseable {
    type ParseOutputType: 'static;
}

pub trait Build<T>: Parseable {
    fn build(pos: TextPosition, types: &mut TypeVec, params: T) -> Self::ParseOutputType;
}

pub trait Parser: Parseable {
    fn is_applicable(stream: &Stream) -> bool;

    fn parse<'a, 'b>(stream: &'a mut Stream, types: &'b mut TypeVec) -> Result<Self::ParseOutputType, CompileError>;
}

pub trait TopLevelParser: Parseable {
    fn parse(steam: &mut Stream) -> Result<Self::ParseOutputType, CompileError>;
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
