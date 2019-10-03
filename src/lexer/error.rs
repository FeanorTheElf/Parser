use super::position::TextPosition;

use std::fmt::{ Display, Formatter, Error };

#[derive(Debug, PartialEq, Eq)]
pub enum ErrorType {
    SyntaxError,
    UndefinedSymbol,
    ShadowedDefinition,
    VariableVoidType,
    DuplicateDefinition
}

#[derive(Debug)]
pub struct CompileError {
    pos: TextPosition,
    msg: String,
    error_type: ErrorType
}

impl CompileError {
    pub fn new(pos: TextPosition, msg: String, error_type: ErrorType) -> Self {
        CompileError {
            pos: pos,
            msg: msg,
            error_type: error_type
        }
    }

    pub fn get_position(&self) -> &TextPosition {
        &self.pos
    }
}

impl Display for CompileError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "Error at {}: {}", self.pos, self.msg)
    }
}