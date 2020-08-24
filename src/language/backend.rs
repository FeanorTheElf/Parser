use super::prelude::*;

#[derive(Debug)]
pub enum OutputError {
    FormatError(std::fmt::Error),
    IOError(std::io::Error),
    UnsupportedCode(TextPosition, String),
}

impl From<std::fmt::Error> for OutputError {
    fn from(error: std::fmt::Error) -> Self {
        OutputError::FormatError(error)
    }
}

impl From<std::io::Error> for OutputError {
    fn from(error: std::io::Error) -> Self {
        OutputError::IOError(error)
    }
}

pub trait Backend<'a> {
    fn print_function_header(&mut self, node: &'a Function) -> Result<(), OutputError>;
    fn enter_block(&mut self, block: &'a Block) -> Result<(), OutputError>;
    fn exit_block(&mut self, block: &'a Block) -> Result<(), OutputError>;
    fn print_parallel_for_header(&mut self, node: &'a ParallelFor) -> Result<(), OutputError>;
    fn print_label(&mut self, node: &'a Label) -> Result<(), OutputError>;
    fn print_goto(&mut self, node: &'a Goto) -> Result<(), OutputError>;
    fn print_if_header(&mut self, node: &'a If) -> Result<(), OutputError>;
    fn print_while_header(&mut self, node: &'a While) -> Result<(), OutputError>;
    fn print_return(&mut self, node: &'a Return) -> Result<(), OutputError>;
    fn print_expression(&mut self, node: &'a Expression) -> Result<(), OutputError>;
    fn print_assignment(&mut self, node: &'a Assignment) -> Result<(), OutputError>;
    fn print_declaration(&mut self, node: &'a LocalVariableDeclaration) -> Result<(), OutputError>;
}

pub trait Printable {
    fn print<'a>(&'a self, printer: &mut (dyn Backend<'a> + 'a)) -> Result<(), OutputError>;
}
