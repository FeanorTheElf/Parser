use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::expression::*;
use super::declaration::*;
use super::writer::*;
use super::CudaContext;

pub trait CudaWritableStatement {
    fn write(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError>;
}

impl CudaWritableStatement for Block {
    fn write(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        unimplemented!()
    }
}

impl CudaWritableStatement for If {
    fn write(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        write!(out, "if (")?;
        self.condition.write_as_value(out, context)?;
        write!(out, ") ")?;
        self.body.write(out, context)?;
        Ok(())
    }
}

impl CudaWritableStatement for While {
    fn write(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        write!(out, "while (")?;
        self.condition.write_as_value(out, context)?;
        write!(out, ") ")?;
        self.body.write(out, context)?;
        Ok(())
    }
}

impl CudaWritableStatement for LocalVariableDeclaration {
    fn write(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        <LocalVariableDeclaration as CudaWritableVariableDeclaration>::write(self, out, context)
    }
}

impl CudaWritableStatement for Return {
    fn write(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        if let Some(return_value) = &self.value {
            write!(out, "return ")?;
            return_value.write_as_value(out, context)?;
        } else {
            write!(out, "return")?;
        }
        write!(out, ";")?;
        Ok(())
    }
}

impl CudaWritableStatement for Assignment {
    fn write(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        self.value.write_assignment_to(&context.calculate_type(&self.assignee), &self.assignee, out, context)
    }
}