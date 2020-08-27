use super::super::language::prelude::*;
use super::super::language::backend::*;
use super::super::transformation::extraction;
use super::ast::*;
use super::expression::*;
use super::statement::*;
use super::kernel_data::*;
use super::function::*;
use super::context::{CudaContext, CudaContextImpl};
use super::super::util::ref_eq::*;

use std::collections::HashSet;

pub struct CudaBackend {

}

impl Backend for CudaBackend {
    
    fn init(&mut self) -> Result<(), OutputError> {
        Ok(())
    }

    fn transform_program(&mut self, program: &mut Program) -> Result<(), OutputError> {
        let mut extractor = extraction::Extractor::new(|_, function| is_generated_with_output_parameter(function.return_type.as_ref().as_ref()));
        extractor.extract_calls_in_program(program);
        return Ok(());
    }

    fn generate<'ast>(&mut self, program: &'ast Program, out: &mut CodeWriter) -> Result<(), OutputError> {
        let mut counter: u32 = 0;
        let mut kernel_id_generator = || { counter += 1; return counter };
        let (mut functions, kernels) = collect_functions(program, &mut kernel_id_generator)?;

        let exported_function = *functions.iter().find(|(func, _)| func.identifier.name.as_str() == "main").unwrap().0;
        functions.get_mut(&exported_function).unwrap().called_from_host = true;

        // TODO: implement less cheap topological sort
        let mut function_kernels_order: Vec<TargetLanguageFunction<'ast>> = Vec::new();
        for (func, _) in functions.iter() {
            function_kernels_order.push(TargetLanguageFunction::Function(*func));
        }
        for (kernel, _) in kernels.iter() {
            function_kernels_order.push(TargetLanguageFunction::Kernel(*kernel));
        }
        let n = functions.len() + kernels.len();
        for _i in 0..n {
            for (func, info) in functions.iter_mut() {
                // update topological sorting
                let current_index = function_kernels_order.iter().position(|f| *f == TargetLanguageFunction::Function(*func)).unwrap();
                let mut new_index = current_index;
                for (index, potential_caller) in function_kernels_order.iter().enumerate().skip(current_index) {
                    if info.called_from.contains(potential_caller) {
                        new_index = index;
                    }
                }
                function_kernels_order.remove(current_index);
                function_kernels_order.insert(new_index, TargetLanguageFunction::Function(*func));

            }
            for (kernel, info) in &kernels {
                let current_index = function_kernels_order.iter().position(|f| *f == TargetLanguageFunction::Kernel(*kernel)).unwrap();
                let mut new_index = current_index;
                for (index, potential_caller) in function_kernels_order.iter().enumerate().skip(current_index) {
                    if info.called_from == *potential_caller {
                        new_index = index;
                    }
                }
                function_kernels_order.remove(current_index);
                function_kernels_order.insert(new_index, TargetLanguageFunction::Kernel(*kernel));
            }
        }

        let mut context = CudaContextImpl::new(program, &functions, &kernels);

        for func in &function_kernels_order {
            match func {
                TargetLanguageFunction::Function(func) => if let Some(f) = gen_function(&*func, &mut context)? {
                    f.write(out)?;
                },
                TargetLanguageFunction::Kernel(kernel) => {
                    gen_kernel(&*kernel, kernels.get(&RefEq::from(&**kernel)).unwrap(), &mut context)?.write(out)?;
                }
            };
        }

        return Ok(());
    }
}