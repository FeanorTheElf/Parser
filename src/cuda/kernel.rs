use super::super::language::prelude::*;
use super::super::analysis::symbol::*;
use super::super::analysis::scope::*;
use super::super::util::ref_eq::*;
use super::super::language::backend::OutputError;
use super::variable::*;
use super::declaration::*;
use super::expression::*;
use super::kernel_data::*;
use super::writer::*;
use super::CudaContext;

use std::collections::HashSet;
use std::iter::FromIterator;

pub trait CudaWritablePFor {
    fn write_definition_as_kernel(&self, kernel: &KernelInfo, out: &mut CodeWriter, context: &mut dyn CudaContext) -> Result<(), OutputError>;
    fn write_definition_as_device_function(&self, kernel: &KernelInfo, out: &mut CodeWriter) -> Result<(), OutputError>;
    fn write_kernel_call(&self, kernel: &KernelInfo, out: &mut CodeWriter) -> Result<(), OutputError>;
}

impl CudaWritablePFor for ParallelFor {

    fn write_definition_as_kernel(&self, kernel: &KernelInfo, out: &mut CodeWriter, context: &mut dyn CudaContext) -> Result<(), OutputError> {
        write!(out, "__global__ void ")?;
        Name::write_kernel(kernel.kernel_name, out)?;
        write!(out, "(")?;

        let kernel_ref = &kernel;
        // Write standard parameters
        out.write_comma_separated(kernel.used_variables.iter().map(|var| move |out: &mut CodeWriter| {
            let declaration = Declaration {
                pos: kernel_ref.pfor.pos().clone(),
                variable: var.get_name().clone(),
                variable_type: var.calc_type().with_view()
            };
            declaration.write_as_param(out)
        }))?;
        write!(out, ", ")?;

        // Write index control parameters
        out.write_comma_separated((0..kernel.pfor.index_variables.len()).map(|dim| move |out: &mut CodeWriter| {
            write!(out, "int ")?;
            Name::write_thread_acc_count(kernel_ref.kernel_name, dim as u32, out)
        }))?;
        write!(out, ", ")?;
        out.write_comma_separated((0..kernel.pfor.index_variables.len()).map(|dim| move |out: &mut CodeWriter| {
            write!(out, "int ")?;
            Name::write_thread_offset(kernel_ref.kernel_name, dim as u32, out)
        }))?;
        write!(out, ") ")?;

        let device_context = context.in_device();

        out.enter_block()?;

        let device_context_ref = &*device_context;
        // Initialize index variables
        out.write_separated(kernel.pfor.index_variables.iter().enumerate().map(|(dim, var)| move |out: &mut CodeWriter| {
            let declaration = CudaVariableDeclaration::new(
                var, 
                Some(CudaExpression::KernelIndexVariableCalculation(kernel_ref.kernel_name, dim as u32, kernel_ref.pfor.index_variables.len() as u32))
            );
            declaration.write(out, device_context_ref)
        }), |out| out.newline().map_err(OutputError::from))?;

        out.exit_block()?;
        return Ok(());
    }

    fn write_definition_as_device_function(&self, kernel:  &KernelInfo, out: &mut CodeWriter) -> Result<(), OutputError> {
        unimplemented!()
    }

    fn write_kernel_call(&self, kernel: &KernelInfo, out: &mut CodeWriter) -> Result<(), OutputError> {
        unimplemented!()
    }
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::Parser;
#[cfg(test)]
use super::context::CudaContextImpl;

#[test]
fn test_write_definition_as_kernel() {
    let pfor = ParallelFor::parse(&mut fragment_lex("
        pfor i: int, with this[i,], in a {
            a[i,] = a[i,] * b;
        }
    ")).unwrap();
    let declaration_a = Declaration {
        pos: super::super::position::NONEXISTING,
        variable: Name::l("a"),
        variable_type: Type::Array(PrimitiveType::Int, 1)
    };
    let kernel_info = KernelInfo {
        called_from: TargetLanguageFunction::Kernel(Ref::from(&pfor)),
        kernel_name: 0,
        pfor: &pfor,
        used_variables: HashSet::from_iter(vec![Ref::from(&declaration_a as &dyn SymbolDefinition)].into_iter())
    };
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let mut counter: u32 = 0;
    let mut context = CudaContextImpl::new(&[], &mut counter);
    pfor.write_definition_as_kernel(&kernel_info, &mut writer, &mut context).unwrap();
    assert_eq!(
"__global__ void kernel_0(int* _a_, unsigned int d0_a_, int d0_thread_acc_count, int d0_thread_offset) {
    int _i_ = static_cast<int>(threadIdx.x) + d0_thread_offset;
}
", output);
}