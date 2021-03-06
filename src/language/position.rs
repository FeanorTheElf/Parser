use std::fmt::{Debug, Display, Error, Formatter};

#[derive(Clone, PartialEq, Eq)]

pub struct TextPosition {
    line: u32,
    column: u32,
}

impl TextPosition {

    pub const BEGIN: TextPosition = TextPosition { line: 0, column: 0 };
    
    pub const NONEXISTING: TextPosition = TextPosition {
        line: u32::max_value(),
        column: u32::max_value(),
    };

    pub fn create(line: u32, column: u32) -> Self {

        TextPosition {
            line: line,
            column: column,
        }
    }

    pub fn column(&self) -> u32 {

        self.column
    }

    pub fn line(&self) -> u32 {

        self.line
    }

    pub fn add_column(&mut self, forward: u32) {

        self.column += forward;
    }

    pub fn next_line(&mut self) {

        self.line += 1;

        self.column = 0;
    }
}

impl Debug for TextPosition {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {

        write!(f, "{}:{}", self.line(), self.column())
    }
}

impl Display for TextPosition {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {

        write!(f, "{}:{}", self.line(), self.column())
    }
}
