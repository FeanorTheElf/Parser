
#[derive(Debug)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Eq)]
pub struct TextPosition {
    line: u32,
    column: u32
}

impl TextPosition {
    pub fn create(line: u32, column: u32) -> Self {
        TextPosition {
            line: line,
            column: column
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