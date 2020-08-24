use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::expression::*;
use super::declaration::*;
use super::writer::*;
use super::CudaContext;

pub trait CudaWritableStatement {
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError>;
}

impl CudaWritableStatement for Block {
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError> {
        unimplemented!()
    }
}

impl CudaWritableStatement for If {
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError> {
        write!(out, "if (")?;
        self.condition.write_as_value(out, context)?;
        write!(out, ") ")?;
        self.body.write(out, context)?;
        Ok(())
    }
}

impl CudaWritableStatement for While {
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError> {
        write!(out, "while (")?;
        self.condition.write_as_value(out, context)?;
        write!(out, ") ")?;
        self.body.write(out, context)?;
        Ok(())
    }
}

impl CudaWritableStatement for LocalVariableDeclaration {
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError> {
        <LocalVariableDeclaration as CudaWritableVariableDeclaration>::write(self, out, context)
    }
}

impl CudaWritableStatement for Return {
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError> {
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
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError> {
        self.value.write_assignment_to(&context.calculate_type(&self.assignee), &self.assignee, out, context)
    }
}

impl CudaWritableStatement for Expression {
    fn write<'stack, 'ast: 'stack>(&self, out: &mut CodeWriter, context: &dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError> {
        match self {
            Expression::Call(call) => {
                let function_type = context.calculate_type(&call.function);
                let (_, return_type) = function_type.expect_callable(self.pos()).unwrap();
                if let Some(return_type) = return_type {
                    match &*return_type.as_ref() {
                        Type::Array(primitive_type, array_dim) => write_temporary_call_array_result_variable(call, *primitive_type, *array_dim, out, context)?,
                        _ => self.write_as_value(out, context)?
                    };
                } else {
                    self.write_as_value(out, context)?;
                    write!(out, ";")?;
                }
                Ok(())
            },
            _ => Err(OutputError::UnsupportedCode(self.pos().clone(), format!("Found expression statement without effect")))
        }
    }
}
