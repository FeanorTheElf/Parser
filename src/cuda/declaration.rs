use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::super::analysis::symbol::*;
use super::expression::*;
use super::writer::*;
use super::CudaContext;
use super::INDEX_TYPE;

pub trait CudaWritableDeclaration {
    fn write_as_param(&self, out: &mut CodeWriter) -> Result<(), OutputError>;
}

impl CudaWritableDeclaration for Declaration {
    fn write_as_param(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self.calc_type() {
            Type::TestType => panic!("TestType not valid"),
            Type::JumpLabel => CompileError::new(self.pos(), format!("Jump label not a valid parameter"), ErrorType::TypeError).throw(),
            Type::Primitive(PrimitiveType::Int) => {
                write!(out, "int ")?;
                self.variable.write_base(out)
            },
            Type::Function(_, _) => Err(OutputError::UnsupportedCode(self.pos().clone(), "Function parameters are not supported".to_owned())),
            Type::Array(_, _) => CompileError::new(self.pos(), format!("Arrays must be passed as views. Consider using &{}", self.variable), ErrorType::ArrayParameterPerValue).throw(),
            Type::View(viewn_type) => match &*viewn_type {
                Type::TestType => panic!("TestType not valid"),
                Type::JumpLabel => CompileError::new(self.pos(), format!("Jump label not a valid parameter"), ErrorType::TypeError).throw(),
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
                Type::View(_) => CompileError::new(self.pos(), format!("Nested views are not allowed"), ErrorType::ViewOnView).throw()
            }
        }
    }
}


pub trait CudaWritableVariableDeclaration {
    fn write(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError>;
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
    fn write(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        match self.declaration.calc_type() {
            Type::Array(PrimitiveType::Int, dim) => {
                write!(out, "int* ")?;
                self.declaration.variable.write_base(out)?;
                if let Some(value) = &self.value {
                    write!(out, " = ")?;
                    value.assert_expression_valid_as_array()?.write_base(out)?;
                }
                write!(out, ";");
                for d in 0..dim {
                    out.newline()?;
                    write!(out, "unsigned int ")?;
                    self.declaration.variable.write_dim(d, out)?;
                    if let Some(value) = &self.value {
                        write!(out, " = ")?;
                        value.assert_expression_valid_as_array()?.write_dim(d, out)?;
                    }
                    write!(out, ";");
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
                    value.write_value_context(out, context)?;
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
    fn write(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        CudaVariableDeclaration {
            declaration: &self.declaration,
            value: self.value.as_ref().map(CudaExpression::Base)
        }.write(out, context)
    }
}