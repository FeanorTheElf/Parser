use super::prelude::*;
use super::print::{ Printer, Printable };

use std::fmt::{ Formatter, Result, Write, Display };

pub struct NazgulDisplay<'a, T: ?Sized + Printable>
{
    node: &'a T
}

impl<'a, T: ?Sized + Printable> Display for NazgulDisplay<'a, T>
{
    fn fmt(&self, f: &mut Formatter) -> Result
    {
        let mut printer: NazgulPrinter = NazgulPrinter {
            result: f,
            indent: 0,
            newline: '\n',
            state: Ok(())
        };
        self.node.print(&mut printer);
        return printer.state;
    }
}

pub fn print_nazgul<T: ?Sized + Printable>(node: &T) -> NazgulDisplay<T>
{
    NazgulDisplay {
        node
    }
}

struct NazgulPrinter<'a, 'b>
{
    result: &'a mut Formatter<'b>,
    indent: usize,
    newline: char,
    state: Result
}

impl<'a, 'b> NazgulPrinter<'a, 'b>
{
    fn indent(&mut self) -> Result
    {
        for _ in 0..self.indent {
            self.result.write_str("    ")?;
        }
        Ok(())
    }

    fn newline(&mut self) -> Result
    {
        self.result.write_char(self.newline)?;
        self.indent()?;
        Ok(())
    }

    fn print_parameters(&mut self, call: &FunctionCall) -> Result
    {
        self.result.write_str("(")?;
        for param in &call.parameters {
            self.print_expr(param)?;
            self.result.write_str(", ")?;
        }
        self.result.write_str(")")?;
        Ok(())
    }

    fn print_expr(&mut self, expr: &Expression) -> Result
    {
        match expr {
            Expression::Call(call) => {
                self.result.write_str("(")?;
                self.print_expr(&call.function)?;
                self.result.write_str(")")?;
                self.print_parameters(call)?;
            },
            Expression::Literal(literal) => {
                literal.value.fmt(&mut self.result)?;
            },
            Expression::Variable(variable) => {
                match &variable.identifier {
                    Identifier::Name(variable_name) => {
                        variable_name.fmt(&mut self.result)?;
                    },
                    Identifier::BuiltIn(builtin_variable) => {
                        builtin_variable.fmt(&mut self.result)?;
                    }
                };
            }
        };
        Ok(())
    }
}

impl<'a, 'b> Printer for NazgulPrinter<'a, 'b>
{
    fn print_function_header(&mut self, node: &Function)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.result.write_str("fn ")?;
            node.identifier.fmt(&mut self.result)?;
            self.result.write_str("(")?;
            for param in &node.params {
                param.name.fmt(&mut self.result)?;
                self.result.write_str(": ")?;
                param.param_type.fmt(&mut self.result)?;
                self.result.write_str(", ")?;
            }
            self.result.write_str(")")?;
            if let Some(return_type) = &node.return_type {
                self.result.write_str(": ")?;
                return_type.fmt(&mut self.result)?;
            }
            self.result.write_str(" ")?;
            Ok(())
        });
    }

    fn enter_block(&mut self) 
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.indent += 1;
            self.result.write_str("{")
        })
    }

    fn exit_block(&mut self)
    {
        self.state = self.state.and_then(|_| {
            self.indent -= 1;
            self.newline()?;
            self.result.write_str("}")?;
            Ok(())
        })
    }

    fn print_if_header(&mut self, node: &If)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.result.write_str("if (")?;
            self.print_expr(&node.condition)?;
            self.result.write_str(") ")?;
            Ok(())
        })
    }

    fn print_while_header(&mut self, node: &While)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.result.write_str("while (")?;
            self.print_expr(&node.condition)?;
            self.result.write_str(") ")?;
            Ok(())
        })
    }

    fn print_return(&mut self, node: &Return)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.result.write_str("return")?;
            if let Some(value) = &node.value {
                self.result.write_str(" ")?;
                self.print_expr(value)?;
            }
            self.result.write_str("; ")?;
            Ok(())
        })
    }

    fn print_expression(&mut self, node: &Expression)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.print_expr(node)?;
            self.result.write_str(";")?;
            Ok(())
        })
    }

    fn print_assignment(&mut self, node: &Assignment)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.print_expr(&node.assignee)?;
            self.result.write_str(" = ")?;
            self.print_expr(&node.value)?;
            self.result.write_str(";")?;
            Ok(())
        })
    }

    fn print_declaration(&mut self, node: &Declaration)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.result.write_str("let ")?;
            node.variable.fmt(&mut self.result)?;
            self.result.write_str(": ")?;
            node.variable_type.fmt(&mut self.result)?;
            if let Some(value) = &node.value {
                self.result.write_str(" = ")?;
                self.print_expr(value)?;
            }
            self.result.write_str(";")?;
            Ok(())
        })
    }
}
