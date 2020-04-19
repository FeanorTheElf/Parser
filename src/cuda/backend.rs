use super::super::language::prelude::*;
use super::super::language::backend::{Backend, OutputError};
use super::function_use_analyzer::FunctionUse;

use std::fmt::Write;
use std::collections::HashMap;

struct OutputThread
{
    out: String,
    indent: String,
    block_level: usize
}

pub struct CudaBackend<'a>
{
    output_threads: Vec<OutputThread>,
    function_uses: HashMap<&'a Name, FunctionUse>,
    create_dim_checks: bool,
    result: Vec<String>,
    kernel_index: usize
}

fn function_not_supported(pos: &TextPosition) -> OutputError
{
    OutputError::UnsupportedCode(pos.clone(), "Cuda backend: type function in this location is not supported".to_owned())
}

fn jump_label_not_allowed(pos: &TextPosition) -> !
{
    CompileError::new(pos, format!("Jump Label is not a valid parameter type"), ErrorType::TypeError).throw()
}

fn panic_test_type() -> !
{
    panic!("")
}

impl<'a> CudaBackend<'a>
{
    fn out(&mut self) -> &mut impl std::fmt::Write
    {
        &mut self.output_threads.last_mut().unwrap().out
    }

    fn increase_indent(&mut self)
    {
    }

    fn decrease_indent(&mut self)
    {
        self.output_threads.last_mut().unwrap().indent.pop();
    }

    fn is_device_context(&self) -> bool
    {
        self.output_threads.len() > 1
    }

    fn print_parameter_function_definition(param: &Declaration) -> Result<Vec<String>, OutputError>
    {
        match &param.variable_type {
            Type::Function(_, _) => Err(function_not_supported(param.pos())),
            Type::JumpLabel => jump_label_not_allowed(param.pos()),
            Type::TestType => panic_test_type(),
            Type::Primitive(PrimitiveType::Int) => Ok(vec![format!("int _{}", param.variable)]),
            Type::Array(PrimitiveType::Int, _) => CompileError::new(param.pos(), format!("Arrays may only be passed per reference"), ErrorType::ArrayParameterPerValue).throw(),
            Type::View(viewn_type) => {
                match &**viewn_type {
                    Type::Primitive(PrimitiveType::Int) => Ok(vec![format!("int * _{}", param.variable)]),
                    Type::Array(PrimitiveType::Int, dim) => {
                        Ok(std::iter::once(format!("int * _{}", param.variable))
                            .chain((0..*dim).map(|i| format!("unsigned dim{}_{}", dim, param.variable)))
                            .collect::<Vec<String>>())
                    },
                    Type::Function(_, _) => Err(function_not_supported(param.pos())),
                    Type::JumpLabel => jump_label_not_allowed(param.pos()),
                    Type::TestType => panic_test_type(),
                    Type::View(_) => CompileError::new(param.pos(), format!("Views on views are not allowed, use single view &... instead"), ErrorType::ViewOnView).throw()
                }
            },
        }
    }

    fn print_all_parameters_function_definition(function: &Function) -> Result<String, OutputError>
    {
        let mut result = Vec::new();
        match &function.return_type {
            Some(Type::Function(_, _)) => Err(function_not_supported(function.pos()))?,
            Some(Type::JumpLabel) => jump_label_not_allowed(function.pos()),
            Some(Type::TestType) => panic_test_type(),
            Some(Type::Primitive(PrimitiveType::Int)) => result.push(format!("int * out_{}", function.identifier)),
            Some(Type::View(_)) => CompileError::new(function.pos(), format!("Views are not allowed as return type"), ErrorType::ViewReturnType).throw(),
            Some(Type::Array(PrimitiveType::Int, dim)) => {
                result.extend(std::iter::once(format!("int ** out_{}", function.identifier)).chain((0..*dim).map(|i| format!("unsigned * outdim{}_{}", i, function.identifier))))
            },
            None => {}
        }
        for param in &function.params {
            result.extend(Self::print_parameter_function_definition(param)?);
        }
        return Ok(result.join(", "));
    }

    fn print_parameter_function_call(&self, expression: &Expression)
    {
        
    }

    fn newline(&mut self) -> Result<(), OutputError>
    {
        let last_entry: &mut OutputThread = self.output_threads.last_mut().unwrap();
        write!(&mut last_entry.out, "\r\n{}", last_entry.indent)?;
        Ok(())
    }
}


impl<'a> Backend for CudaBackend<'a>
{
    fn print_function_header(&mut self, node: &Function) -> Result<(), OutputError>
    {
        let function_use = self.function_uses.get(&node.identifier).unwrap();
        let device_modifier = if function_use.device_called { "__device__" } else { "" };
        let host_modifier = if function_use.host_called { "__host__" } else { "" };
        write!(self.out(), "{} {} void _{}({})", device_modifier, host_modifier, node.identifier, Self::print_all_parameters_function_definition(node)?)?;
        Ok(())
    }

    fn enter_block(&mut self) -> Result<(), OutputError>
    {
        write!(self.out(), "{{")?;
        let mut current_out = self.output_threads.last_mut().unwrap();
        current_out.indent.push('\t');
        current_out.block_level += 1;
        Ok(())
    }

    fn exit_block(&mut self) -> Result<(), OutputError>
    {
        let mut current_out = self.output_threads.last_mut().unwrap();
        current_out.indent.pop();
        if current_out.block_level > 0 {
            current_out.block_level -= 1;
            self.newline()?;
            return Ok(());
        } else {
            // close current output thread
            let out = self.output_threads.pop().unwrap();
            self.result.push(out.out);
            return Ok(());
        }
    }
    
    fn print_parallel_for_header(&mut self, node: &ParallelFor) -> Result<(), OutputError>
    {
        self.enter_block()?;
        self.newline()?;
        write!(self.out(), "dim3 griddim_ = dim3();")?;
        self.newline()?;
        write!(self.out(), "dim3 blockdim_ = dim3();")?;
        self.newline()?;
        write!(self.out(), "kernel_{}<<<griddim_, blockdim_>>>({});", 0, 0)?;
        Ok(())
    }
    
    fn print_label(&mut self, node: &Label) -> Result<(), OutputError>
    {
        Ok(())
    }
    
    fn print_goto(&mut self, node: &Goto) -> Result<(), OutputError>
    {
        Ok(())
    }
    
    fn print_if_header(&mut self, node: &If) -> Result<(), OutputError>
    {
        Ok(())
    }
    
    fn print_while_header(&mut self, node: &While) -> Result<(), OutputError>
    {
        Ok(())
    }
    
    fn print_return(&mut self, node: &Return) -> Result<(), OutputError>
    {
        Ok(())
    }
    
    fn print_expression(&mut self, node: &Expression) -> Result<(), OutputError>
    {
        Ok(())
    }
    
    fn print_assignment(&mut self, node: &Assignment) -> Result<(), OutputError>
    {
        Ok(())
    }
    
    fn print_declaration(&mut self, node: &LocalVariableDeclaration) -> Result<(), OutputError>
    {
        Ok(())
    }
}