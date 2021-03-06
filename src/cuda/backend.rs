use super::super::language::compiler::*;
use super::super::language::prelude::*;
use super::super::transformation::extraction;
use super::super::analysis::export::*;
use super::super::util::ref_eq::*;
use super::ast::*;
use super::context::{CudaContext, CudaContextImpl};
use super::expression::*;
use super::function::*;
use super::kernel_data::*;
use std::collections::HashMap;

pub struct CudaBackend {
    header_template: Option<String>,
}

fn topological_sort<'ast>(
    function_kernels_order: &mut Vec<TargetLanguageFunction<'ast>>,
    functions: &HashMap<Ptr<'ast, Function>, FunctionInfo<'ast>>,
    kernels: &HashMap<Ptr<'ast, ParallelFor>, KernelInfo<'ast>>,
) {

    for _i in 0..(functions.len() + kernels.len()) {

        for (func, info) in functions {

            // update topological sorting
            let current_index = function_kernels_order
                .iter()
                .position(|f| *f == TargetLanguageFunction::Function(*func))
                .unwrap();

            let mut new_index = current_index;

            for (index, potential_caller) in function_kernels_order
                .iter()
                .enumerate()
                .skip(current_index)
            {

                if info.called_from.contains(potential_caller) {

                    new_index = index;
                }
            }

            function_kernels_order.remove(current_index);

            function_kernels_order.insert(new_index, TargetLanguageFunction::Function(*func));
        }

        for (kernel, info) in kernels {

            let current_index = function_kernels_order
                .iter()
                .position(|f| *f == TargetLanguageFunction::Kernel(*kernel))
                .unwrap();

            let mut new_index = current_index;

            for (index, potential_caller) in function_kernels_order
                .iter()
                .enumerate()
                .skip(current_index)
            {

                if info.called_from == *potential_caller {

                    new_index = index;
                }
            }

            function_kernels_order.remove(current_index);

            function_kernels_order.insert(new_index, TargetLanguageFunction::Kernel(*kernel));
        }
    }
}

impl CudaBackend {
    pub fn new() -> Self {

        CudaBackend {
            header_template: None,
        }
    }
}

impl Compiler for CudaBackend {
    fn init(&mut self) -> Result<(), OutputError> {
        self.header_template = Some(std::fs::read_to_string("./cuda/template.cuh")?);
        Ok(())
    }

    fn transform_program(&mut self, program: &mut Program) -> Result<(), OutputError> {
        let (items, prog_lifetime) = program.work();
        let mut extractor = extraction::Extractor::new(|_, function| {
            is_generated_with_output_parameter(function.get_type(prog_lifetime).return_type(prog_lifetime))
        });
        extractor.extract_calls_in_program((items, prog_lifetime));
        return Ok(());
    }

    fn generate<'ast>(
        &mut self,
        program: &'ast Program,
        out: &mut CodeWriter,
    ) -> Result<(), OutputError> {

        let mut counter: u32 = 0;
        let mut kernel_id_generator = || {
            counter += 1;
            return counter;
        };
        let (mut functions, kernels) = collect_functions(program, &mut kernel_id_generator)?;

        for exported_function in get_functions_to_export(&program.items) {
            functions
                .get_mut(&Ptr::from(exported_function))
                .unwrap()
                .called_from_host = true;
        }


        // TODO: implement less cheap topological sort
        let mut function_kernels_order: Vec<TargetLanguageFunction<'ast>> = Vec::new();

        for (func, _) in functions.iter() {

            function_kernels_order.push(TargetLanguageFunction::Function(*func));
        }

        for (kernel, _) in kernels.iter() {

            function_kernels_order.push(TargetLanguageFunction::Kernel(*kernel));
        }

        topological_sort(&mut function_kernels_order, &functions, &kernels);

        for func in &function_kernels_order {

            if let TargetLanguageFunction::Function(func) = func {

                let called_from_host = functions
                    .get(&RefEq::from(&**func))
                    .unwrap()
                    .called_from
                    .iter()
                    .filter_map(|f| match f {
                        TargetLanguageFunction::Kernel(_) => None,
                        TargetLanguageFunction::Function(f) => Some(f),
                    })
                    .any(|f| functions.get(&RefEq::from(&**f)).unwrap().called_from_host);

                let called_from_device = functions
                    .get(&RefEq::from(&**func))
                    .unwrap()
                    .called_from
                    .iter()
                    .any(|f| match f {
                        TargetLanguageFunction::Kernel(_) => true,
                        TargetLanguageFunction::Function(f) => {
                            functions
                                .get(&RefEq::from(&**f))
                                .unwrap()
                                .called_from_device
                        }
                    });

                let mut info = functions.get_mut(&RefEq::from(&**func)).unwrap();

                info.called_from_host |= called_from_host;

                info.called_from_device |= called_from_device;
            }
        }

        function_kernels_order.reverse();

        let mut context = CudaContextImpl::new(program, &functions, &kernels);
        assert!(context.get_scopes().is_global_scope());

        write!(out, "{}", self.header_template.as_ref().unwrap())?;

        let mut funcs = function_kernels_order.iter().peekable();

        while let Some(func) = funcs.next() {

            match func {
                TargetLanguageFunction::Function(func) => {

                    if let Some(f) = gen_function(&**func, &mut context)? {

                        f.write(out)?;
                    }
                }
                TargetLanguageFunction::Kernel(kernel) => {

                    gen_kernel(
                        &**kernel,
                        kernels.get(&RefEq::from(&**kernel)).unwrap(),
                        &mut context,
                    )?
                    .write(out)?;
                }
            };
        }

        return Ok(());
    }
}

#[cfg(test)]
use super::super::lexer::lexer::lex_str;
#[cfg(test)]
use super::super::parser::TopLevelParser;
#[cfg(test)]
use std::iter::FromIterator;

#[test]
fn test_topological_sort() {

    let mut program = Program::parse(&mut lex_str("
    fn foo(a: int,): int {
        return bar(a,) + 1;
    }

    fn bar(a: int,): int {
        return a + 1;
    }

    fn main(x: int,): int {
        return foo(bar(x,),) + 1;
    }
    "))
    .unwrap();

    let mut output = "".to_owned();

    let mut target = StringWriter::new(&mut output);

    let mut writer = CodeWriter::new(&mut target);

    let mut backend = CudaBackend::new();

    backend.init().unwrap();

    backend.transform_program(&mut program).unwrap();

    backend.generate(&program, &mut writer).unwrap();

    assert_eq!(
        backend.header_template.unwrap()
            + "

__host__ inline int bar_(int a_) {
    return a_ + 1;
}

__host__ inline int foo_(int a_) {
    return bar_(a_) + 1;
}

__host__ inline int main_(int x_) {
    return foo_(bar_(x_)) + 1;
}",
        output
    );
}
