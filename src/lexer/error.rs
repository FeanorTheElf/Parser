use super::position::TextPosition;

use std::fmt::{ Display, Formatter, Error };

#[derive(Debug)]
pub struct CompileError {
    pos: TextPosition,
    msg: String
}

impl CompileError {
    pub fn new(pos: TextPosition, msg: String) -> Self {
        CompileError {
            pos: pos,
            msg: msg
        }
    }
}

impl Display for CompileError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "Error at {}: {}", self.pos, self.msg)
    }
}