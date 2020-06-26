use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::super::analysis::symbol::*;
use super::writer::*;

const VAR_PREFIX: &'static str = "_";
const DIM_PREFIX: &'static str = "d";
const INDEX_TYPE: &'static str = "unsigned int";

pub trait CudaWritableVariable {
    fn write_base(&self, out: &mut CodeWriter) -> Result<(), OutputError>;
    fn write_dim(&self, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError>;
}

impl CudaWritableVariable for Name {
    fn write_base(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        if self.id != 0 {
            write!(out, "{}{}_{}", VAR_PREFIX, self.name, self.id).map_err(OutputError::from)
        } else {
            write!(out, "{}{}_", VAR_PREFIX, self.name).map_err(OutputError::from)
        }
    }

    fn write_dim(&self, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
        if self.id != 0 {
            write!(out, "{}{}{}{}_{}", DIM_PREFIX, dim, VAR_PREFIX, self.name, self.id).map_err(OutputError::from)
        } else {
            write!(out, "{}{}{}{}_", DIM_PREFIX, dim, VAR_PREFIX, self.name).map_err(OutputError::from)
        }
    }
}

pub trait CudaWritableDeclaration {
    fn write_as_param(&self, f: &mut CodeWriter) -> Result<(), OutputError>;
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