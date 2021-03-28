use super::language::error::CompileError;
use super::language::position::TextPosition;
use super::lexer::tokens::Stream;

pub trait Parseable {
    type ParseOutputType: 'static;
}

pub struct ParserContext {

}

impl ParserContext {
    
    pub fn new() -> ParserContext {
        ParserContext {}
    }
}

pub trait Build<T>: Parseable {
    fn build(pos: TextPosition, context: &mut ParserContext, params: T) -> Self::ParseOutputType;
}

pub trait Parser: Parseable {
    fn is_applicable(stream: &Stream) -> bool;

    fn parse<'a, 'b>(stream: &'a mut Stream, context: &'b mut ParserContext) -> Result<Self::ParseOutputType, CompileError>;
}

pub trait TopLevelParser: Parseable {
    fn parse(steam: &mut Stream) -> Result<Self::ParseOutputType, CompileError>;
}

#[macro_use]
pub mod parser_gen;
pub mod parser;
