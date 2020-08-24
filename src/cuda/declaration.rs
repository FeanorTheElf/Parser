use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::super::analysis::symbol::*;
use super::super::analysis::type_error::*;
use super::writer::*;
use super::expression::*;
use super::variable::*;
use super::CudaContext;
use super::INDEX_TYPE;

pub trait CudaWritableDeclaration {
    fn write_as_param(&self, out: &mut CodeWriter) -> Result<(), OutputError>;
}

impl CudaWritableDeclaration for Declaration {
    fn write_as_param(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        let pos = self.pos();
        match self.calc_type() {
            Type::TestType => error_test_type(pos),
            Type::JumpLabel => error_jump_label_var_type(pos).throw(),
            Type::Primitive(PrimitiveType::Int) => {
                write!(out, "int ")?;
                self.variable.write_base(out)
            },
            Type::Function(_, _) => Err(OutputError::UnsupportedCode(self.pos().clone(), "Function parameters are not supported".to_owned())),
            Type::Array(_, _) => error_array_value_parameter(pos).throw(),
            Type::View(viewn_type) => match &*viewn_type {
                Type::TestType => error_test_type(pos),
                Type::JumpLabel => error_jump_label_var_type(pos).throw(),
                Type::Primitive(PrimitiveType::Int) => {
                    write!(out, "int* ")?;
                    self.variable.write_base(out)
                },
                Type::Function(_, _) => Err(OutputError::UnsupportedCode(self.pos().clone(), "Function parameters are not supported".to_owned())),
                Type::Array(PrimitiveType::Int, dim) => {
                    write!(out, "int* ")?;
                    self.variable.write_base(out)?;
                    for d in 0..*dim {
                        write!(out, ", ")?;
                        write!(out, "{}", INDEX_TYPE)?;
                        write!(out, " ")?;
                        self.variable.write_dim(d, out)?;
                    }
                    Ok(())
                },
                Type::View(_) => error_nested_view(pos).throw()
            }
        }
    }
}

pub trait CudaWritableVariableDeclaration {
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError>;
}

pub struct CudaVariableDeclaration<'a> {
    declaration: &'a Declaration,
    value: Option<CudaExpression<'a>>
}

impl<'a> CudaVariableDeclaration<'a> {
    pub fn new(decl: &'a Declaration, value: Option<CudaExpression<'a>>) -> Self {
        CudaVariableDeclaration {
            declaration: decl,
            value: value
        }
    }
}

impl<'a> CudaWritableVariableDeclaration for CudaVariableDeclaration<'a> {
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError> {
        match self.declaration.calc_type() {
            Type::Array(PrimitiveType::Int, dim) => {
                write!(out, "int* ")?;
                self.declaration.variable.write_base(out)?;
                write!(out, ";")?;
                for d in 0..dim {
                    out.newline()?;
                    write!(out, "unsigned int ")?;
                    self.declaration.variable.write_dim(d, out)?;
                    write!(out, ";")?;
                }
                if let Some(value) = &self.value {
                    value.write_assignment_to(&self.declaration.variable_type, &Expression::Variable(Variable { 
                        identifier: Identifier::Name(self.declaration.variable.clone()),
                        pos: self.declaration.pos().clone()
                    }), out, context)?;
                }
                Ok(())
            },
            Type::Function(_, _) => Err(OutputError::UnsupportedCode(self.declaration.pos().clone(), format!("Local variables with function types are not supported"))),
            Type::JumpLabel => CompileError::new(self.declaration.pos(), format!("Jump labels as local variable types are illegal"), ErrorType::TypeError).throw(),
            Type::Primitive(PrimitiveType::Int) => {
                write!(out, "int ")?;
                self.declaration.variable.write_base(out)?;
                if let Some(value) = &self.value {
                    write!(out, " = ")?;
                    value.write_as_value(out, context)?;
                }
                write!(out, ";")?;
                Ok(())
            },
            Type::TestType => panic!("TestType"),
            Type::View(_) => Err(OutputError::UnsupportedCode(self.declaration.pos().clone(), format!("Local variables with view types are not supported")))
        }
    }
}

impl CudaWritableVariableDeclaration for LocalVariableDeclaration {
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError> {
        CudaVariableDeclaration {
            declaration: &self.declaration,
            value: self.value.as_ref().map(CudaExpression::Base)
        }.write(out, context)
    }
}