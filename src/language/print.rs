use super::prelude::*;

pub trait Printer {
    fn print_function_header(&mut self, node: &Function);
    fn enter_block(&mut self);
    fn exit_block(&mut self);
    fn print_parallel_for_header(&mut self, node: &ParallelFor);
    fn print_label(&mut self, node: &Label);
    fn print_goto(&mut self, node: &Goto);
    fn print_if_header(&mut self, node: &If);
    fn print_while_header(&mut self, node: &While);
    fn print_return(&mut self, node: &Return);
    fn print_expression(&mut self, node: &Expression);
    fn print_assignment(&mut self, node: &Assignment);
    fn print_declaration(&mut self, node: &LocalVariableDeclaration);
}

pub trait Printable {
    fn print<'a>(&self, printer: &mut (dyn Printer + 'a));
}
