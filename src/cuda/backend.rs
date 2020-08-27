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

use std::collections::{HashSet, HashMap};

pub struct CudaBackend {

}

fn topological_sort<'ast>(function_kernels_order: &mut Vec<TargetLanguageFunction<'ast>>, functions: &HashMap<Ref<'ast, Function>, FunctionInfo<'ast>>, kernels: &HashMap<Ref<'ast, ParallelFor>, KernelInfo<'ast>>) {
    for _i in 0..(functions.len() + kernels.len()) {
        for (func, info) in functions {
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
        for (kernel, info) in kernels {
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
        topological_sort(&mut function_kernels_order, &functions, &kernels);
        
        let mut transitive_called_from_host = HashSet::new();
        transitive_called_from_host.insert(TargetLanguageFunction::Function(exported_function));
        let mut transitive_called_from_device = HashSet::new();
        transitive_called_from_device.extend(kernels.iter().map(|(kernel, _)| TargetLanguageFunction::Kernel(*kernel)));
        for (func, info) in functions.iter_mut() {
            if !info.called_from_host && info.called_from.iter().any(|caller| transitive_called_from_host.contains(caller)) {
                info.called_from_host = true;
                transitive_called_from_host.insert(TargetLanguageFunction::Function(*func));
            }
            if !info.called_from_device && info.called_from.iter().any(|caller| transitive_called_from_device.contains(caller)) {
                info.called_from_device = true;
                transitive_called_from_device.insert(TargetLanguageFunction::Function(*func));
            }
        }

        function_kernels_order.reverse();

        let mut context = CudaContextImpl::new(program, &functions, &kernels);

        let mut funcs = function_kernels_order.iter().peekable();
        while let Some(func) = funcs.next() {
            match func {
                TargetLanguageFunction::Function(func) => if let Some(f) = gen_function(&**func, &mut context)? {
                    f.write(out)?;
                },
                TargetLanguageFunction::Kernel(kernel) => {
                    gen_kernel(&**kernel, kernels.get(&RefEq::from(&**kernel)).unwrap(), &mut context)?.write(out)?;
                }
            };
            if funcs.peek().is_some() {
                out.newline()?;
                out.newline()?;
            }
        }

        return Ok(());
    }
}

#[cfg(test)]
use std::iter::FromIterator;
#[cfg(test)]
use super::super::util::ref_eq::*;
#[cfg(test)]
use super::super::lexer::lexer::lex;
#[cfg(test)]
use super::super::parser::Parser;

#[test]
fn test_topological_sort() {
    let mut program = Program::parse(&mut lex("
    fn foo(a: int,): int {
        return bar(a,) + 1;
    }

    fn bar(a: int,): int {
        return a + 1;
    }

    fn main(x: int,): int {
        return foo(bar(x,),) + 1;
    }
    ")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let mut backend = CudaBackend{};
    backend.init().unwrap();
    backend.transform_program(&mut program).unwrap();
    backend.generate(&program, &mut writer).unwrap();
    assert_eq!(
"__host__ int bar_(int a_) {
    return a_ + 1;
}

__host__ int foo_(int a_) {
    return bar_(a_) + 1;
}

__host__ int main_(int x_) {
    return foo_(bar_(x_)) + 1;
}", output);
}