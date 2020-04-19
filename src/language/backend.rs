use super::prelude::*;

#[derive(Debug)]
pub enum OutputError {
    FormatError(std::fmt::Error),
    UnsupportedCode(TextPosition, String)
}

impl From<std::fmt::Error> for OutputError
{
    fn from(error: std::fmt::Error) -> Self
    {
        OutputError::FormatError(error)
    }
}

pub trait Backend {
    fn print_function_header(&mut self, node: &Function) -> Result<(), OutputError>;
    fn enter_block(&mut self) -> Result<(), OutputError>;
    fn exit_block(&mut self) -> Result<(), OutputError>;
    fn print_parallel_for_header(&mut self, node: &ParallelFor) -> Result<(), OutputError>;
    fn print_label(&mut self, node: &Label) -> Result<(), OutputError>;
    fn print_goto(&mut self, node: &Goto) -> Result<(), OutputError>;
    fn print_if_header(&mut self, node: &If) -> Result<(), OutputError>;
    fn print_while_header(&mut self, node: &While) -> Result<(), OutputError>;
    fn print_return(&mut self, node: &Return) -> Result<(), OutputError>;
    fn print_expression(&mut self, node: &Expression) -> Result<(), OutputError>;
    fn print_assignment(&mut self, node: &Assignment) -> Result<(), OutputError>;
    fn print_declaration(&mut self, node: &LocalVariableDeclaration) -> Result<(), OutputError>;
}

pub trait Printable {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError>;
}
