use super::super::language::prelude::*;
use super::super::analysis::symbol::*;
use super::super::language::backend::OutputError;
use super::ast::*;
use super::statement::*;
use super::kernel_data::*;
use super::writer::*;
use super::CudaContext;

fn gen_kernel<'stack, 'ast: 'stack>(pfor: &ParallelFor, kernel: &KernelInfo, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<CudaKernel, OutputError> {
    let name = CudaIdentifier::Kernel(kernel.kernel_name);

    let standard_parameters = kernel.used_variables.iter().flat_map(|var| gen_variables(pfor.pos(), var.get_name(), &var.calc_type()));
    let grid_size_variables = (0..kernel.pfor.index_variables.len()).map(|dim| CudaIdentifier::ThreadGridSizeVar(kernel.kernel_name, dim as u32));
    let grid_size_parameters = std::iter::repeat(CudaType {
        base: CudaPrimitiveType::Index,
        constant: true,
        ptr_count: 0
    }).zip(grid_size_variables.clone());
    let grid_offset_variables = (0..kernel.pfor.index_variables.len()).map(|dim| CudaIdentifier::ThreadGridOffsetVar(kernel.kernel_name, dim as u32));
    let grid_offset_parameters = std::iter::repeat(CudaType {
        base: CudaPrimitiveType::Int,
        constant: true,
        ptr_count: 0
    }).zip(grid_offset_variables.clone());

    context.set_device();

    let thread_index = CudaExpression::Sum(vec![
        (AddSub::Plus, CudaExpression::Identifier(CudaIdentifier::ThreadIdxX)),
        (AddSub::Plus, CudaExpression::Product(vec![
            (MulDiv::Multiply, CudaExpression::Identifier(CudaIdentifier::BlockIdxX)),
            (MulDiv::Multiply, CudaExpression::Identifier(CudaIdentifier::BlockDimX))
        ]))
    ]);
    let grid_size_variables_vec: Vec<_> = grid_size_variables.map(CudaExpression::Identifier).collect();
    let grid_offset_variables_vec: Vec<_> = grid_offset_variables.map(CudaExpression::Identifier).collect();
    let init_index_vars = kernel.pfor.index_variables.iter().enumerate().map(|(dim, var)| CudaVarDeclaration {
        value: Some(CudaExpression::MultiDimIndexCalculation(dim as u32, Box::new(thread_index.clone()), grid_size_variables_vec.clone(), grid_offset_variables_vec.clone())),
        var: CudaIdentifier::ValueVar(var.get_name().clone()),
        var_type: CudaType {
            base: CudaPrimitiveType::Int,
            constant: true,
            ptr_count: 0
        }
    }).map(|v| Box::new(v) as Box<dyn CudaStatement>);

    // TODO: generate body
    let body = CudaIf {
        body: CudaBlock {
            statements: vec![]
        },
        cond: CudaExpression::Comparison(Cmp::Ls, 
            Box::new(thread_index.clone()), 
            Box::new(CudaExpression::Identifier(CudaIdentifier::ThreadGridSizeVar(kernel.kernel_name, 0))))
    };

    context.set_host();

    return Ok(CudaKernel {
        name: name,
        params: standard_parameters.chain(grid_size_parameters).chain(grid_offset_parameters).collect::<Vec<_>>(),
        body: CudaBlock {
            statements: init_index_vars.chain(std::iter::once(Box::new(body) as Box<dyn CudaStatement>)).collect()
        }
    })
}

#[cfg(test)]
use std::collections::HashSet;
#[cfg(test)]
use std::iter::FromIterator;
#[cfg(test)]
use super::super::util::ref_eq::*;
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
    gen_kernel(&pfor, &kernel_info, &mut context).unwrap().write(&mut writer).unwrap();
    assert_eq!(
"__global__ void kernel0(int* a_, const unsigned int a_d0, const unsigned int kernel0d0, const int kernel0o0) {
    const int i_ = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) + kernel0o0;
    if (threadIdx.x + blockIdx.x * blockDim.x < kernel0d0) {
        
    };
    
}", output);
}