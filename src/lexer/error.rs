use super::position::TextPosition;

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