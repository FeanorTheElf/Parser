use super::prelude::*;
use super::print::{ Printer, Printable };

use std::fmt::{ Formatter, Result, Write, Display };

pub struct DebugDisplayWrapper<'a, T: ?Sized + Printable>
{
    node: &'a T
}

impl<'a, T: ?Sized + Printable> Display for DebugDisplayWrapper<'a, T>
{
    fn fmt(&self, f: &mut Formatter) -> Result
    {
        let mut printer: DebugPrinter = DebugPrinter {
            result: f,
            indent: 0,
            newline: '\n',
            state: Ok(())
        };
        self.node.print(&mut printer);
        return printer.state;
    }
}

pub fn print_debug<T: ?Sized + Printable>(node: &T) -> DebugDisplayWrapper<T>
{
    DebugDisplayWrapper {
        node
    }
}

struct DebugPrinter<'a, 'b>
{
    result: &'a mut Formatter<'b>,
    indent: usize,
    newline: char,
    state: Result
}

impl<'a, 'b> DebugPrinter<'a, 'b>
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

impl<'a, 'b> Printer for DebugPrinter<'a, 'b>
{
    fn print_function_header(&mut self, node: &Function)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.result.write_str("fn ")?;
            node.identifier.fmt(&mut self.result)?;
            self.result.write_str("(")?;
            for parameter_declaration in &node.params {
                parameter_declaration.variable.fmt(&mut self.result)?;
                self.result.write_str(": ")?;
                parameter_declaration.variable_type.fmt(&mut self.result)?;
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

    fn print_label(&mut self, node: &Label)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.result.write_str("@")?;
            node.label.fmt(&mut self.result)?;
            Ok(())
        })
    }

    fn print_goto(&mut self, node: &Goto)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.result.write_str("goto ")?;
            node.target.fmt(&mut self.result)?;
            self.result.write_str(";")?;
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

    fn print_declaration(&mut self, node: &LocalVariableDeclaration)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.result.write_str("let ")?;
            node.declaration.variable.fmt(&mut self.result)?;
            self.result.write_str(": ")?;
            node.declaration.variable_type.fmt(&mut self.result)?;
            if let Some(value) = &node.value {
                self.result.write_str(" = ")?;
                self.print_expr(value)?;
            }
            self.result.write_str(";")?;
            Ok(())
        })
    }

    fn print_parallel_for_header(&mut self, node: &ParallelFor)
    {
        self.state = self.state.and_then(|_| {
            self.newline()?;
            self.result.write_str("pfor ")?;
            for index_variable in &node.index_variables {
                index_variable.variable.fmt(&mut self.result)?;
                self.result.write_str(": ")?;
                index_variable.variable_type.fmt(&mut self.result)?;
                self.result.write_str(", ")?;
            }
            for array_access_pattern in &node.access_pattern {
                self.result.write_str("with ")?;
                for entry_access in &array_access_pattern.entry_accesses {
                    self.result.write_str("this[")?;
                    for index in entry_access.get_indices() {
                        self.print_expr(index)?;
                        self.result.write_str(", ")?;
                    }
                    self.result.write_str("]")?;
                    if let Some(alias) = &entry_access.alias {
                        self.result.write_str(" as ")?;
                        alias.fmt(&mut self.result)?;
                    }
                    self.result.write_str(", ")?;
                }
                self.result.write_str("in ")?;
                self.print_expr(&array_access_pattern.array)?;
            }
            Ok(())
        })
    }
}
