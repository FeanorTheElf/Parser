use super::position::TextPosition;

use std::fmt::{Display, Error, Formatter};

const INTERNAL_ERROR: &'static str =
    "Compiler error should have been detected earlier, did all checkers run?";

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ErrorType {
    SyntaxError,
    UndefinedSymbol,
    ShadowedDefinition,
    VariableVoidType,
    DuplicateDefinition,
    IllegalArrayBaseType,
    IncorrectIdentifier,
    IllegalPForIndexExpression,
    PForAccessCollision,
    VariableRequired,
    TypeError,
    ArrayParameterPerValue,
    ViewOnView,
    ViewReturnType,
    ArrayParameterByValue,
    RValueAssignment,
}

#[derive(Debug, Clone)]
pub struct CompileError {
    pos: TextPosition,
    msg: String,
    error_type: ErrorType,
}

impl CompileError {
    pub fn new(pos: &TextPosition, msg: String, error_type: ErrorType) -> Self {
        CompileError {
            pos: pos.clone(),
            msg: msg,
            error_type: error_type,
        }
    }

    pub fn get_position(&self) -> &TextPosition {
        &self.pos
    }

    pub fn throw(self) -> ! {
        panic!(format!("Error at {}: {}", self.pos, self.msg))
    }
}

impl Display for CompileError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "Error at {}: {}", self.pos, self.msg)
    }
}

pub trait InternalErrorConvertable<T> {
    fn internal_error(self) -> T;
}

impl<T> InternalErrorConvertable<T> for Result<T, CompileError> {
    fn internal_error(self) -> T {
        self.expect(INTERNAL_ERROR)
    }
}
