use super::super::analysis::symbol::*;
use super::super::analysis::type_error::*;
use super::super::language::compiler::*;
use super::super::language::prelude::*;
use super::ast::*;
use super::context::CudaContext;
use super::expression::*;
use super::kernel_data::*;
use super::statement::*;
use feanor_la::prelude::*;
use feanor_la::rat::r64;

fn gen_if<'stack, 'ast: 'stack>(
    statement: &'ast If,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    Ok(Box::new(CudaIf {
        body: gen_block(&statement.body, context)?,
        cond: gen_expression(&statement.condition, context)?,
    }))
}

fn gen_while<'stack, 'ast: 'stack>(
    statement: &'ast While,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    Ok(Box::new(CudaWhile {
        body: gen_block(&statement.body, context)?,
        cond: gen_expression(&statement.condition, context)?,
    }))
}

pub fn gen_kernel<'c, 'stack, 'ast: 'stack>(
    pfor: &'ast ParallelFor,
    kernel: &'c KernelInfo<'ast>,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<CudaKernel, OutputError> {

    let name = CudaIdentifier::Kernel(kernel.kernel_name);
    let ast_lifetime = context.ast_lifetime();

    let standard_parameters = kernel.used_variables.iter().flat_map(|var| {

        let var_type = Type::View(var.calc_type(ast_lifetime).expect_array(pfor.pos()).internal_error().clone().reference_view());

        let parameter_variables = gen_variables(pfor.pos(), var.get_name(), &var_type);

        return parameter_variables.collect::<Vec<_>>().into_iter();
    });

    let grid_size_variables = (0..kernel.pfor.index_variables.len())
        .map(|dim| CudaIdentifier::ThreadGridSizeVar(kernel.kernel_name, dim as u32));

    let grid_size_parameters = std::iter::repeat(CudaType {
        base: CudaPrimitiveType::Index,
        constant: true,
        ptr: false,
        owned: false
    })
    .zip(grid_size_variables.clone());

    let grid_offset_variables = (0..kernel.pfor.index_variables.len())
        .map(|dim| CudaIdentifier::ThreadGridOffsetVar(kernel.kernel_name, dim as u32));

    let grid_offset_parameters = std::iter::repeat(CudaType {
        base: CudaPrimitiveType::Int,
        owned: false,
        ptr: false,
        constant: true
    },)
    .zip(grid_offset_variables.clone());

    context.set_device();

    context.enter_scope(kernel);

    let thread_index = CudaExpression::Sum(vec![
        (
            AddSub::Plus,
            CudaExpression::Identifier(CudaIdentifier::ThreadIdxX),
        ),
        (
            AddSub::Plus,
            CudaExpression::Product(vec![
                (
                    MulDiv::Multiply,
                    CudaExpression::Identifier(CudaIdentifier::BlockIdxX),
                ),
                (
                    MulDiv::Multiply,
                    CudaExpression::Identifier(CudaIdentifier::BlockDimX),
                ),
            ]),
        ),
    ]);

    let grid_size_variables_vec: Vec<_> = grid_size_variables
        .map(CudaExpression::Identifier)
        .collect();

    let grid_offset_variables_vec: Vec<_> = grid_offset_variables
        .map(CudaExpression::Identifier)
        .collect();

    let init_index_vars = kernel
        .pfor
        .index_variables
        .iter()
        .enumerate()
        .map(|(dim, var)| CudaVarDeclaration {
            value: Some(CudaExpression::MultiDimIndexCalculation(
                dim as u32,
                Box::new(thread_index.clone()),
                grid_size_variables_vec.clone(),
                grid_offset_variables_vec.clone(),
            )),
            var: CudaIdentifier::ValueVar(var.get_name().clone()),
            var_type: CudaType {
                base: CudaPrimitiveType::Int,
                owned: false,
                ptr: false,
                constant: true
            },
        })
        .map(|v| Box::new(v) as Box<dyn CudaStatement>);

    context.enter_scope(pfor);

    let body = gen_block(&pfor.body, context)?;

    context.exit_scope();

    let body = CudaIf {
        body: body,
        cond: CudaExpression::Comparison(
            Cmp::Ls,
            Box::new(thread_index.clone()),
            Box::new(CudaExpression::Identifier(
                CudaIdentifier::ThreadGridSizeVar(kernel.kernel_name, 0),
            )),
        ),
    };

    context.exit_scope();

    context.set_host();

    return Ok(CudaKernel {
        name: name,
        params: standard_parameters
            .chain(grid_size_parameters)
            .chain(grid_offset_parameters)
            .collect::<Vec<_>>(),
        body: CudaBlock {
            statements: init_index_vars
                .chain(std::iter::once(Box::new(body) as Box<dyn CudaStatement>))
                .collect(),
        },
    });
}

fn for_powerset<'a, T, F, E>(a: &'a [T], b: &'a [T], mut f: F) -> Result<(), E>
where
    F: for<'b> FnMut(&'b [&'b T]) -> Result<(), E>,
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

fn gen_kernel_call<'stack, 'ast: 'stack>(
    pfor: &ParallelFor,
    kernel: &KernelInfo,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    assert!(pfor.index_variables.len() > 0);

    let mut statements: Vec<Box<dyn CudaStatement>> = Vec::new();

    let mut local_array_id: usize = 0;

    let mut array_dim_counts = Vec::new();

    // First: calculate the size of the arrays, not the standard postfix size products
    for access_pattern in &pfor.access_pattern {

        let access_pattern_array_type = context.calculate_type(&access_pattern.array);

        let array_type = access_pattern_array_type.expect_indexable(access_pattern.pos()).internal_error();

        array_dim_counts.push(array_type.dimension);

        let size_exprs =
            gen_simple_expr_array_size(&access_pattern.array, &access_pattern_array_type).collect::<Vec<_>>();

        assert_eq!(size_exprs.len(), array_type.dimension);

        for d in 0..array_type.dimension {

            let value = if d + 1 == array_type.dimension {

                size_exprs[d as usize].1.clone()
            } else {

                CudaExpression::Product(vec![
                    (MulDiv::Multiply, size_exprs[d as usize].1.clone()),
                    (MulDiv::Divide, size_exprs[d as usize + 1].1.clone()),
                ])
            };

            statements.push(Box::new(CudaVarDeclaration {
                value: Some(value),
                var_type: CudaType {
                    base: CudaPrimitiveType::Index,
                    owned: false,
                    ptr: false,
                    constant: true
                },
                var: CudaIdentifier::TmpArrayShapeVar(local_array_id as u32, d as u32),
            }));
        }

        local_array_id += 1;
    }

    // Second: calculate the index variable values corresponding to the corners of the (convex) array index set
    let mut edge_coordinates = Vec::new();

    edge_coordinates.resize(pfor.index_variables.len(), vec![]);

    local_array_id = 0;

    for access_pattern in &pfor.access_pattern {

        let array_size_vars = (0..array_dim_counts[local_array_id])
            .map(|d| {

                CudaExpression::Identifier(CudaIdentifier::TmpArrayShapeVar(
                    local_array_id as u32,
                    d as u32,
                ))
            })
            .collect::<Vec<_>>();

        let zeros = (0..array_dim_counts[local_array_id])
            .map(|d| CudaExpression::FloatLiteral(0.))
            .collect::<Vec<_>>();

        for entry_access in &access_pattern.entry_accesses {

            let access_matrix: Matrix<r64> = Matrix::from(
                entry_access
                    .get_transformation_matrix(&pfor.index_variables)
                    .unwrap()
                    .as_ref(),
            );

            let translation = access_matrix.get((.., 0..=0));

            assert_eq!(ACCESS_MATRIX_AFFINE_COLUMN, 0);

            let inv_access_matrix = access_matrix.get((.., 1..)).invert().unwrap();

            let index_var_translation: Matrix<CudaExpression> =
                Matrix::from_nocopy(inv_access_matrix.as_ref() * translation);

            let inv_access_matrix_expr: Matrix<CudaExpression> =
                Matrix::from_nocopy(inv_access_matrix);

            debug_assert_eq!(inv_access_matrix_expr.rows(), pfor.index_variables.len());

            for_powerset::<_, _, ()>(&array_size_vars[..], &zeros[..], |corner_point_data| {

                let corner_point = Vector::new(
                    corner_point_data
                        .iter()
                        .map(|x| *x)
                        .map(CudaExpression::clone)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                );

                let index_var_space_corner_point = (inv_access_matrix_expr.as_ref() * corner_point)
                    .as_ref()
                    - index_var_translation.as_ref();

                for (i, (expr, _, _)) in index_var_space_corner_point.into_iter().enumerate() {

                    edge_coordinates[i].push(expr);
                }

                return Ok(());
            })
            .unwrap();
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
                base: CudaPrimitiveType::Int,
                owned: false,
                ptr: false,
                constant: true
            },
            value: Some(CudaExpression::Round(Box::new(CudaExpression::Min(
                coordinates.clone(),
            )))),
        }));

        statements.push(Box::new(CudaVarDeclaration {
            var: size.clone(),
            var_type: CudaType {
                base: CudaPrimitiveType::Index,
                owned: false,
                ptr: false,
                constant: true
            },
            value: Some(
                CudaExpression::Round(Box::new(CudaExpression::Max(coordinates)))
                    - CudaExpression::Identifier(offset),
            ),
        }));
    }

    // Fourth: collect the parameters
    let standard_parameters = kernel.used_variables.iter().flat_map(|var| {

        gen_variables_for_view(pfor.pos(), var.get_name(), &var.calc_type(context.ast_lifetime()))
            .map(|(_, expr)| expr)
            .collect::<Vec<_>>()
            .into_iter()
    });

    let grid_size_variables = (0..kernel.pfor.index_variables.len())
        .map(|dim| CudaIdentifier::ThreadGridSizeVar(kernel.kernel_name, dim as u32))
        .map(CudaExpression::Identifier);

    let grid_offset_variables = (0..kernel.pfor.index_variables.len())
        .map(|dim| CudaIdentifier::ThreadGridOffsetVar(kernel.kernel_name, dim as u32))
        .map(CudaExpression::Identifier);

    let parameters = standard_parameters
        .chain(grid_size_variables)
        .chain(grid_offset_variables);

    // Now we have calculated the size parameters
    // Fifth: generate kernel call
    let last_dim = pfor.index_variables.len() - 1;

    let block_size = CudaExpression::IntLiteral(256);

    let grid_size = CudaExpression::Product(
        (0..last_dim)
            .map(|d| {

                (
                    MulDiv::Multiply,
                    CudaExpression::Identifier(CudaIdentifier::ThreadGridSizeVar(
                        kernel.kernel_name,
                        d as u32,
                    )),
                )
            })
            .collect::<Vec<_>>(),
    ) * CudaExpression::IndexFloorDiv(
        Box::new(CudaExpression::Identifier(
            CudaIdentifier::ThreadGridSizeVar(kernel.kernel_name, last_dim as u32),
        )),
        Box::new(block_size.clone()),
    );

    statements.push(Box::new(CudaKernelCall {
        name: CudaIdentifier::Kernel(kernel.kernel_name),
        grid_size: grid_size,
        block_size: block_size,
        shared_mem_size: CudaExpression::IntLiteral(0),
        params: parameters.collect::<Vec<_>>(),
    }));

    return Ok(Box::new(CudaBlock {
        statements: statements,
    }));
}

pub fn gen_block<'stack, 'ast: 'stack>(
    block: &'ast Block,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<CudaBlock, OutputError> {

    let mut result_statements: Vec<Box<dyn CudaStatement>> = Vec::new();

    context.enter_scope(block);

    let scope_levels = context.get_scopes().get_scope_levels();

    for statement in &block.statements {

        if let Some(expr) = statement.dynamic().downcast_ref::<Expression>() {

            result_statements
                .push(Box::new(gen_expression(expr, context)?) as Box<dyn CudaStatement>);
        } else if let Some(if_statement) = statement.dynamic().downcast_ref::<If>() {

            result_statements.push(gen_if(if_statement, context)?);
        } else if let Some(while_statement) = statement.dynamic().downcast_ref::<While>() {

            result_statements.push(gen_while(while_statement, context)?);
        } else if let Some(return_statement) = statement.dynamic().downcast_ref::<Return>() {

            result_statements.push(gen_return(return_statement, context)?);
        } else if let Some(block) = statement.dynamic().downcast_ref::<Block>() {

            result_statements.push(Box::new(gen_block(block, context)?));
        } else if let Some(declaration) = statement
            .dynamic()
            .downcast_ref::<LocalVariableDeclaration>()
        {

            for v in gen_localvardef(declaration, context) {

                result_statements.push(v?);
            }
        } else if let Some(assignment) = statement.dynamic().downcast_ref::<Assignment>() {

            result_statements.push(gen_assignment(assignment, context)?);
        } else if let Some(label) = statement.dynamic().downcast_ref::<Label>() {

            result_statements.push(gen_label(label, context)?);
        } else if let Some(goto) = statement.dynamic().downcast_ref::<Goto>() {

            result_statements.push(gen_goto(goto, context)?);
        } else if let Some(pfor) = statement.dynamic().downcast_ref::<ParallelFor>() {

            let pfor_data = context.get_pfor_data(pfor);

            result_statements.push(gen_kernel_call(pfor, pfor_data, context)?);
        } else {

            panic!("Unknown statement type: {:?}", statement);
        }
    }

    debug_assert_eq!(context.get_scopes().get_scope_levels(), scope_levels);

    context.exit_scope();

    Ok(CudaBlock {
        statements: result_statements,
    })
}

fn gen_implemented_function<'data, 'ast: 'data>(
    function: &'ast Function,
    body: &'ast Block,
    context: &mut dyn CudaContext<'data, 'ast>,
) -> Result<CudaFunction, OutputError> {

    context.set_current_function(function);

    context.enter_scope(function);
    let scope_levels = context.get_scopes().get_scope_levels();

    let function_info = context.get_function_data(function);
    let ast_lifetime = context.ast_lifetime();
    let standard_params = function
        .params
        .iter()
        .flat_map(|p| gen_variables(p.pos(), &p.variable, &*ast_lifetime.cast(p.variable_type).borrow()).collect::<Vec<_>>().into_iter());

    let result = if is_generated_with_output_parameter(function.return_type.map(|t| ast_lifetime.cast(t))) {

        let params = if let Some(return_type) = &function.return_type {
            standard_params
                .chain(gen_output_parameter_declaration(
                    function.pos(),
                    ast_lifetime.cast(*return_type).borrow().expect_array(function.pos()).internal_error(),
                ))
                .collect::<Vec<_>>()
        } else {
            standard_params.collect::<Vec<_>>()
        };

        Ok(CudaFunction {
            device: function_info.called_from_device,
            host: function_info.called_from_host,
            name: CudaIdentifier::ValueVar(function.identifier.clone()),
            params: params,
            return_type: CudaType {
                base: CudaPrimitiveType::Void,
                owned: false,
                ptr: false,
                constant: false
            },
            body: gen_block(body, context)?,
        })
    } else {
        let ast_lifetime = context.ast_lifetime();
        let return_type = if let Some(ty) = &function.return_type.map(|x| ast_lifetime.cast(x)) {
            match &*ty.borrow() {
                Type::Array(arr) => {
                    assert_eq!(arr.dimension, 0);
                    CudaType {
                        base: gen_primitive_type(arr.base),
                        owned: false,
                        ptr: false,
                        constant: false
                    }
                },
                Type::Function(_) => {
                    return Err(OutputError::UnsupportedCode(
                        function.pos().clone(),
                        format!("Functions that return functions are not supported in cuda backend"),
                    ))
                },
                Type::JumpLabel => error_jump_label_var_type(function.pos()).throw(),
                Type::TestType => error_test_type(function.pos()),
                Type::View(_viewn) => error_return_view(function.pos()).throw(),
            }
        } else {
            CudaType {
                base: CudaPrimitiveType::Void,
                owned: false,
                ptr: false,
                constant: false
            }
        };

        Ok(CudaFunction {
            device: function_info.called_from_device,
            host: function_info.called_from_host,
            name: CudaIdentifier::ValueVar(function.identifier.clone()),
            params: standard_params.collect::<Vec<_>>(),
            return_type: return_type,
            body: gen_block(body, context)?,
        })
    };

    debug_assert_eq!(context.get_scopes().get_scope_levels(), scope_levels);
    context.exit_scope();

    return result;
}

pub fn gen_function<'stack, 'ast: 'stack>(
    function: &'ast Function,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Option<CudaFunction>, OutputError> {

    debug_assert!(context.get_scopes().is_global_scope());
    let result = if let Some(body) = &function.body {

        Some(gen_implemented_function(function, body, context)).transpose()
    } else {

        Ok(None)
    };
    debug_assert!(context.get_scopes().is_global_scope());
    return result;
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::Parser;
#[cfg(test)]
use super::super::util::ref_eq::*;
#[cfg(test)]
use super::context::CudaContextImpl;
#[cfg(test)]
use std::collections::BTreeSet;
#[cfg(test)]
use std::iter::FromIterator;

#[test]

fn test_gen_kernel() {
    let mut types = TypeVec::new();
    let pfor = ParallelFor::parse(&mut fragment_lex(
        "
        pfor i: int, with this[i,], in a {
            a[i,] = a[i,] * b;
        }
    "), &mut types)
    .unwrap();

    let declaration_a = (Name::l("a"), Type::Array(ArrayType { base: PrimitiveType::Int, dimension: 1 }));

    let declaration_b = (Name::l("b"), Type::Array(ArrayType { base: PrimitiveType::Int, dimension: 0 }));

    let defs = [declaration_a, declaration_b];

    let kernel_info = KernelInfo {
        called_from: TargetLanguageFunction::Kernel(Ref::from(&pfor)),
        kernel_name: 0,
        pfor: &pfor,
        used_variables: BTreeSet::from_iter(
            defs.iter()
                .map(|d| d as &dyn SymbolDefinition)
                .map(SortByNameSymbolDefinition::from),
        ),
    };

    let program = Program { items: vec![], types: types };

    let mut output = "".to_owned();

    let mut target = StringWriter::new(&mut output);

    let mut writer = CodeWriter::new(&mut target);

    let mut context: Box<dyn CudaContext> =
        Box::new(CudaContextImpl::build_with_leak(&program).unwrap());

    context.enter_scope(&defs[..]);

    gen_kernel(&pfor, &kernel_info, &mut *context)
        .unwrap()
        .write(&mut writer)
        .unwrap();

    assert_eq!("

__global__ inline void kernel0(int* a_, unsigned int a_d0, int* b_, const unsigned int kernel0d0, const int kernel0o0) {
    const int i_ = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) + kernel0o0;
    if (threadIdx.x + blockIdx.x * blockDim.x < kernel0d0) {
        a_[i_] = a_[i_] * b_;
    };
}", output);
}

#[test]

fn test_gen_kernel_call() {

    let pfor = ParallelFor::parse(&mut fragment_lex(
        "
        pfor i: int, with this[i,], in a {
            a[i,] = a[i,] * b;
        }
    "), &mut TypeVec::new())
    .unwrap();

    let mut program = Program { items: vec![], types: TypeVec::new() };

    let declaration_a = (Name::l("a"), Type::Array(ArrayType { base: PrimitiveType::Int, dimension: 1 }));

    let declaration_b = (Name::l("b"), Type::Array(ArrayType { base: PrimitiveType::Int, dimension: 0 }));

    let defs = [declaration_a, declaration_b];

    let kernel_info = KernelInfo {
        called_from: TargetLanguageFunction::Kernel(Ref::from(&pfor)),
        kernel_name: 0,
        pfor: &pfor,
        used_variables: BTreeSet::from_iter(
            defs.iter()
                .map(|d| d as &dyn SymbolDefinition)
                .map(SortByNameSymbolDefinition::from),
        ),
    };


    let mut output = "".to_owned();

    let mut target = StringWriter::new(&mut output);

    let mut writer = CodeWriter::new(&mut target);

    let mut context: Box<dyn CudaContext> =
        Box::new(CudaContextImpl::build_with_leak(&program).unwrap());

    context.enter_scope(&defs[..]);

    gen_kernel_call(&pfor, &kernel_info, &mut *context)
        .unwrap()
        .write(&mut writer)
        .unwrap();

    assert_eq!("{
    const unsigned int array0shape0 = a_d0;
    const int kernel0o0 = round(min(array0shape0, 0));
    const unsigned int kernel0d0 = round(max(array0shape0, 0)) - kernel0o0;
    kernel0 <<< dim3((kernel0d0 - 1) / 256 + 1), dim3(256), 0 >>> (a_, a_d0, &b_, kernel0d0, kernel0o0);
}", output);
}

#[test]

fn test_gen_kernel_call_complex() {

    let pfor = ParallelFor::parse(&mut fragment_lex(
        "
        pfor i: int, j: int, with this[2 * i + j, j,], this[2 * i + j + 1, j,], in a {
            a[2 * i + j + 1, j,] = a[2 * i + j, j,];
        }
    "), &mut TypeVec::new())
    .unwrap();

    let declaration_a = (Name::l("a"), Type::Array(ArrayType { base: PrimitiveType::Int, dimension: 2 }));

    let defs = [declaration_a];

    let kernel_info = KernelInfo {
        called_from: TargetLanguageFunction::Kernel(Ref::from(&pfor)),
        kernel_name: 0,
        pfor: &pfor,
        used_variables: BTreeSet::from_iter(
            defs.iter()
                .map(|d| d as &dyn SymbolDefinition)
                .map(SortByNameSymbolDefinition::from),
        ),
    };

    let program = Program { items: vec![], types: TypeVec::new() };

    let mut output = "".to_owned();

    let mut target = StringWriter::new(&mut output);

    let mut writer = CodeWriter::new(&mut target);

    let mut context: Box<dyn CudaContext> =
        Box::new(CudaContextImpl::build_with_leak(&program).unwrap());

    context.enter_scope(&defs[..]);

    gen_kernel_call(&pfor, &kernel_info, &mut *context)
        .unwrap()
        .write(&mut writer)
        .unwrap();

    assert_eq!("{
    const unsigned int array0shape0 = a_d0 / a_d1;
    const unsigned int array0shape1 = a_d1;
    const int kernel0o0 = round(min((1./2.) * array0shape0 + (-1./2.) * array0shape1, (-1./2.) * array0shape1, (1./2.) * array0shape0, 0, (1./2.) * array0shape0 + (-1./2.) * array0shape1 - 1./2., (-1./2.) * array0shape1 - 1./2., (1./2.) * array0shape0 - 1./2., 0 - 1./2.));
    const unsigned int kernel0d0 = round(max((1./2.) * array0shape0 + (-1./2.) * array0shape1, (-1./2.) * array0shape1, (1./2.) * array0shape0, 0, (1./2.) * array0shape0 + (-1./2.) * array0shape1 - 1./2., (-1./2.) * array0shape1 - 1./2., (1./2.) * array0shape0 - 1./2., 0 - 1./2.)) - kernel0o0;
    const int kernel0o1 = round(min(array0shape1, array0shape1, 0, 0, array0shape1, array0shape1, 0, 0));
    const unsigned int kernel0d1 = round(max(array0shape1, array0shape1, 0, 0, array0shape1, array0shape1, 0, 0)) - kernel0o1;
    kernel0 <<< dim3(kernel0d0 * ((kernel0d1 - 1) / 256 + 1)), dim3(256), 0 >>> (a_, a_d0, a_d1, kernel0d0, kernel0d1, kernel0o0, kernel0o1);
}", output);
}
