use super::super::language::prelude::*;
use super::super::analysis::symbol::*;
use super::super::language::backend::OutputError;
use super::ast::*;
use super::expression::*;
use super::statement::*;
use super::kernel_data::*;
use super::context::CudaContext;
use feanor_la::prelude::*;
use feanor_la::rat::r64;

fn gen_if<'stack, 'ast: 'stack>(statement: &If, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    Ok(Box::new(CudaIf {
        body: CudaBlock {
            statements: vec![]
        },
        cond: gen_expression(&statement.condition, context)?
    }))
}

fn gen_while<'stack, 'ast: 'stack>(statement: &While, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    Ok(Box::new(CudaWhile {
        body: CudaBlock {
            statements: vec![]
        },
        cond: gen_expression(&statement.condition, context)?
    }))
}

fn gen_kernel<'stack, 'ast: 'stack>(pfor: &ParallelFor, kernel: &KernelInfo, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<CudaKernel, OutputError> {
    let name = CudaIdentifier::Kernel(kernel.kernel_name);

    let standard_parameters = kernel.used_variables.iter().flat_map(|var| {
        let var_type = var.calc_type();
        let passed_parameters = gen_variables_as_view(pfor.pos(), var.get_name(), &var_type);
        let parameter_variables = gen_variables(pfor.pos(), var.get_name(), &var_type);
        return passed_parameters.map(|(ty, _)| ty).zip(parameter_variables.map(|(_, name)| name)).collect::<Vec<_>>().into_iter();
    });
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

fn for_powerset<'a, T, F, E>(a: &'a [T], b: &'a [T], mut f: F) -> Result<(), E>
    where F: for<'b> FnMut(&'b [&'b T]) -> Result<(), E>
{
    assert_eq!(a.len(), b.len());
    assert!(a.len() < 64);
    let mut set = Vec::new();
    set.reserve(a.len());
    for i in 0..a.len() {
        set.push(&a[i]);
    }
    f(&set[..])?;
    for counter in 1..(1 << a.len()) {
        let trailing_zeros = u64::trailing_zeros(counter) as usize;
        for i in 0..trailing_zeros {
            set[i] = &a[i];
        }
        set[trailing_zeros] = &b[trailing_zeros];
        f(&set[..])?;
    }
    Ok(())
}

fn gen_kernel_call<'stack, 'ast: 'stack>(pfor: &ParallelFor, kernel: &KernelInfo, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    assert!(pfor.index_variables.len() > 0);

    let mut statements: Vec<Box<dyn CudaStatement>> = Vec::new();
    let mut local_array_id: usize = 0;
    let mut array_dim_counts = Vec::new();

    // First: calculate the size of the arrays, not the standard postfix size products
    for access_pattern in &pfor.access_pattern {
        let array_type = context.calculate_type(&access_pattern.array);
        let (_, dim) = expect_array_type(&array_type);
        array_dim_counts.push(dim);
        let size_exprs = gen_array_size_exprs(&access_pattern.array, &array_type).collect::<Vec<_>>();
        assert_eq!(size_exprs.len() as u32, dim);
        for d in 0..dim {
            let value = if d + 1 == dim {
                size_exprs[d as usize].1.clone()
            } else {
                CudaExpression::Product(vec![
                    (MulDiv::Multiply, size_exprs[d as usize].1.clone()),
                    (MulDiv::Divide, size_exprs[d as usize + 1].1.clone())
                ])
            };
            statements.push(Box::new(CudaVarDeclaration {
                value: Some(value),
                var_type: CudaType {
                    ptr_count: 0,
                    constant: true,
                    base: CudaPrimitiveType::Index
                },
                var: CudaIdentifier::TmpArrayShapeVar(local_array_id as u32, d)
            }));
        }
        local_array_id += 1;
    }

    // Second: calculate the index variable values corresponding to the corners of the (convex) array index set 
    let mut edge_coordinates = Vec::new();
    edge_coordinates.resize(pfor.index_variables.len(), vec![]);
    local_array_id = 0;
    for access_pattern in &pfor.access_pattern {
        let array_size_vars = (0..array_dim_counts[local_array_id]).map(|d| CudaExpression::Identifier(CudaIdentifier::TmpArrayShapeVar(local_array_id as u32, d))).collect::<Vec<_>>();
        let zeros = (0..array_dim_counts[local_array_id]).map(|d| CudaExpression::FloatLiteral(0.)).collect::<Vec<_>>();
        for entry_access in &access_pattern.entry_accesses {
            let access_matrix: Matrix<r64> = Matrix::from(entry_access.get_transformation_matrix(&pfor.index_variables).unwrap().as_ref());
            let translation = access_matrix.get((.., 0..=0));
            assert_eq!(ACCESS_MATRIX_AFFINE_COLUMN, 0);

            let inv_access_matrix = access_matrix.get((.., 1..)).invert().unwrap();
            let index_var_translation: Matrix<CudaExpression> = Matrix::from_nocopy(inv_access_matrix.as_ref() * translation);
            let inv_access_matrix_expr: Matrix<CudaExpression> = Matrix::from_nocopy(inv_access_matrix);
            debug_assert_eq!(inv_access_matrix_expr.rows(), pfor.index_variables.len());

            for_powerset::<_, _, ()>(&array_size_vars[..], &zeros[..], |corner_point_data| {
                let corner_point = Vector::new(corner_point_data.iter().map(|x| *x).map(CudaExpression::clone).collect::<Vec<_>>().into_boxed_slice());
                let index_var_space_corner_point = (inv_access_matrix_expr.as_ref() * corner_point).as_ref() - index_var_translation.as_ref();
                for (i, (expr, _, _)) in index_var_space_corner_point.into_iter().enumerate() {
                    edge_coordinates[i].push(expr);
                }
                return Ok(());
            }).unwrap();
        }
        local_array_id += 1;
    }

    // Third: the minimal, rounded index corner coordinate is the offset, and the difference between min/max is the number of threads
    for (i, coordinates) in edge_coordinates.into_iter().enumerate() {
        let offset = CudaIdentifier::ThreadGridOffsetVar(kernel.kernel_name, i as u32);
        let size = CudaIdentifier::ThreadGridSizeVar(kernel.kernel_name, i as u32);
        statements.push(Box::new(CudaVarDeclaration {
            var: offset.clone(),
            var_type: CudaType {
                base: CudaPrimitiveType::Index,
                ptr_count: 0,
                constant: true
            },
            value: Some(CudaExpression::Round(Box::new(CudaExpression::Min(coordinates.clone()))))
        }));
        statements.push(Box::new(CudaVarDeclaration {
            var: size.clone(),
            var_type: CudaType {
                base: CudaPrimitiveType::Index,
                ptr_count: 0,
                constant: true
            },
            value: Some(CudaExpression::Round(Box::new(CudaExpression::Max(coordinates))) - CudaExpression::Identifier(offset))
        }));
    }

    // Fourth: collect the parameters
    let standard_parameters = kernel.used_variables.iter().flat_map(|var| gen_variables_as_view(pfor.pos(), var.get_name(), &var.calc_type()).map(|(_, expr)| expr).collect::<Vec<_>>().into_iter());
    let grid_size_variables = (0..kernel.pfor.index_variables.len()).map(|dim| CudaIdentifier::ThreadGridSizeVar(kernel.kernel_name, dim as u32)).map(CudaExpression::Identifier);
    let grid_offset_variables = (0..kernel.pfor.index_variables.len()).map(|dim| CudaIdentifier::ThreadGridOffsetVar(kernel.kernel_name, dim as u32)).map(CudaExpression::Identifier);
    let parameters = standard_parameters.chain(grid_size_variables).chain(grid_offset_variables);

    // Now we have calculated the size parameters
    // Fifth: generate kernel call
    let last_dim = pfor.index_variables.len() - 1;
    let block_size = CudaExpression::IntLiteral(256);
    let grid_size = CudaExpression::Product(
        (0..last_dim).map(|d| (MulDiv::Multiply, CudaExpression::Identifier(CudaIdentifier::ThreadGridSizeVar(kernel.kernel_name, d as u32)))).collect::<Vec<_>>()
    ) * CudaExpression::IndexFloorDiv(
        Box::new(CudaExpression::Identifier(CudaIdentifier::ThreadGridSizeVar(kernel.kernel_name, last_dim as u32))), 
        Box::new(block_size.clone())
    );
    
    statements.push(Box::new(CudaKernelCall {
        name: CudaIdentifier::Kernel(kernel.kernel_name), 
        grid_size: grid_size, 
        block_size: block_size, 
        shared_mem_size: CudaExpression::IntLiteral(0), 
        params: parameters.collect::<Vec<_>>()
    }));

    return Ok(Box::new(CudaBlock {
        statements: statements
    }));
}

pub fn gen_block<'stack, 'ast: 'stack>(block: &'ast Block, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    let mut result_statements: Vec<Box<dyn CudaStatement>> = Vec::new();
    context.enter_scope(block);
    for statement in &block.statements {
        if let Some(expr) = statement.dynamic().downcast_ref::<Expression>() {
            result_statements.push(Box::new(gen_expression(expr, context)?) as Box<dyn CudaStatement>);
        } else if let Some(if_statement) = statement.dynamic().downcast_ref::<If>() {
            result_statements.push(gen_if(if_statement, context)?);
        } else if let Some(while_statement) = statement.dynamic().downcast_ref::<While>() {
            result_statements.push(gen_while(while_statement, context)?);
        } else if let Some(return_statement) = statement.dynamic().downcast_ref::<Return>() {
            result_statements.push(gen_return(return_statement, context)?);
        } else if let Some(block) = statement.dynamic().downcast_ref::<Block>() {
            result_statements.push(gen_block(block, context)?);
        } else if let Some(declaration) = statement.dynamic().downcast_ref::<LocalVariableDeclaration>() {
            for v in gen_localvardef(declaration, context) {
                result_statements.push(v?);
            }
        } else if let Some(assignment) = statement.dynamic().downcast_ref::<Assignment>() {
            result_statements.push(gen_assignment(assignment, context)?);
        } else if let Some(label) = statement.dynamic().downcast_ref::<Label>() {
            result_statements.push(gen_label(label, context)?);
        } else if let Some(goto) = statement.dynamic().downcast_ref::<Goto>() {
            result_statements.push(gen_goto(goto, context)?);
        } else {
            panic!("Unknown statement type: {:?}", statement);
        }
    }
    context.exit_scope();
    unimplemented!()
}

#[cfg(test)]
use std::collections::BTreeSet;
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
#[cfg(test)]
use super::writer::*;

#[test]
fn test_write_definition_as_kernel() {
    let pfor = ParallelFor::parse(&mut fragment_lex("
        pfor i: int, with this[i,], in a {
            a[i,] = a[i,] * b;
        }
    ")).unwrap();
    let declaration_a = (Name::l("a"), Type::Array(PrimitiveType::Int, 1));
    let declaration_b = (Name::l("b"), Type::Primitive(PrimitiveType::Int));
    let defs = [declaration_a, declaration_b];
    let kernel_info = KernelInfo {
        called_from: TargetLanguageFunction::Kernel(Ref::from(&pfor)),
        kernel_name: 0,
        pfor: &pfor,
        used_variables: BTreeSet::from_iter(defs.iter().map(|d| d as &dyn SymbolDefinition).map(SortByNameSymbolDefinition::from))
    };
    let program = Program { items: vec![] };
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::build_for_program(&program).unwrap());
    context.enter_scope(&defs[..]);
    gen_kernel(&pfor, &kernel_info, &mut *context).unwrap().write(&mut writer).unwrap();
    assert_eq!(
"__global__ void kernel0(int* a_, const unsigned int a_d0, int* b_, const unsigned int kernel0d0, const int kernel0o0) {
    const int i_ = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) + kernel0o0;
    if (threadIdx.x + blockIdx.x * blockDim.x < kernel0d0) {
        
    };
}", output);
}

#[test]
fn test_write_kernel_call() {
    let pfor = ParallelFor::parse(&mut fragment_lex("
        pfor i: int, with this[i,], in a {
            a[i,] = a[i,] * b;
        }
    ")).unwrap();
    let declaration_a = (Name::l("a"), Type::Array(PrimitiveType::Int, 1));
    let declaration_b = (Name::l("b"), Type::Primitive(PrimitiveType::Int));
    let defs = [declaration_a, declaration_b];
    let kernel_info = KernelInfo {
        called_from: TargetLanguageFunction::Kernel(Ref::from(&pfor)),
        kernel_name: 0,
        pfor: &pfor,
        used_variables: BTreeSet::from_iter(defs.iter().map(|d| d as &dyn SymbolDefinition).map(SortByNameSymbolDefinition::from))
    };
    let program = Program { items: vec![] };
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::build_for_program(&program).unwrap());
    context.enter_scope(&defs[..]);
    gen_kernel_call(&pfor, &kernel_info, &mut *context).unwrap().write(&mut writer).unwrap();
    assert_eq!("{
    const unsigned int array0shape0 = a_d0;
    const unsigned int kernel0o0 = round(min(array0shape0, 0));
    const unsigned int kernel0d0 = round(max(array0shape0, 0)) - kernel0o0;
    kernel0 <<< dim3(static_cast<int>((kernel0d0 - 1) / 256 + 1), dim3(256), 0 >>> (a_, a_d0, &b_, kernel0d0, kernel0o0);
}", output);
}