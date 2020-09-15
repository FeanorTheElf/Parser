use super::super::analysis::type_error::*;
use super::super::language::compiler::*;
use super::super::language::prelude::*;
use super::ast::*;
use super::context::CudaContext;

fn error_call_dynamic_expression(pos: &TextPosition) -> OutputError {

    OutputError::UnsupportedCode(
        pos.clone(),
        format!("Calling dynamic expressions is not supported by the Cuda Backend"),
    )
}

pub fn is_mul_var_type(ty: &Type) -> bool {

    match ty {
        Type::Array(arr) => arr.dimension > 0,
        Type::View(view) => view.base.dimension > 0,
        _ => false,
    }
}

pub fn is_generated_with_output_parameter<T>(return_type: Option<T>) -> bool
    where T: std::ops::Deref<Target = Type>
{
    return_type
        .map(|ty| is_mul_var_type(&*ty))
        .unwrap_or(false)
}

pub fn gen_primitive_type(ty: PrimitiveType) -> CudaPrimitiveType {
    assert_eq!(ty, PrimitiveType::Int);
    CudaPrimitiveType::Int
}

pub fn gen_array_size<'a>(
    pos: &TextPosition,
    array_variable: &'a Name,
    ty: &Type,
) -> impl 'a + Iterator<Item = (CudaType, CudaIdentifier)> {

    let without_view = ty.clone().without_view();
    let arr = without_view.expect_array(pos).internal_error();

    (0..arr.dimension).map(move |d| {

        (
            CudaType {
                ptr: false,
                constant: false,
                owned: false,
                base: CudaPrimitiveType::Index,
            },
            CudaIdentifier::ArraySizeVar(array_variable.clone(), d as u32),
        )
    })
}

pub fn gen_simple_expr_array_size<'a>(
    expr: &'a Expression,
    ty: &Type,
) -> impl 'a + Iterator<Item = (CudaType, CudaExpression)> {

    match expr {
        Expression::Call(_) => panic!("gen_simple_expr_array_size() got a call expression, but calls yielding arrays should be extracted"),
        Expression::Literal(_) => panic!("currently, have no array literals"),
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(op) => panic!("currently, have no builting identifiers that have array type, got {}", op),
            Identifier::Name(name) => gen_array_size(expr.pos(), name, ty).map(|(ty, var)| (ty, CudaExpression::Identifier(var)))
        }
    }
}

/// Generates the cuda variable containing the data from the given variable
pub fn gen_variable(pos: &TextPosition, name: &Name, ty: &Type) -> (CudaType, CudaIdentifier) {

    match ty {
        Type::Array(arr) => (
            CudaType {
                base: gen_primitive_type(arr.base),
                owned: arr.dimension > 0,
                constant: false,
                ptr: false
            },
            CudaIdentifier::ValueVar(name.clone()),
        ),
        Type::Function(_) => panic!("Cannot represent a function in cuda variables"),
        Type::JumpLabel => panic!("Cannot represent a jump label in cuda variables"),
        Type::TestType => error_test_type(pos),
        Type::View(view) => (
            CudaType {
                base: gen_primitive_type(view.base.base),
                owned: false,
                constant: false,
                ptr: true
            },
            CudaIdentifier::ValueVar(name.clone()),
        )
    }
}

pub fn gen_simple_expr<'a>(expr: &'a Expression, ty: &Type) -> (CudaType, CudaExpression) {

    match expr {
        Expression::Call(_) if is_mul_var_type(ty) => panic!("gen_simple_expr() got function call expression yielding array, should have been extracted earlier"),
        Expression::Call(_) => panic!("gen_simple_expr() got call expression, to generate expressions recursivly use gen_expression() instead"),
        Expression::Literal(_) => panic!("currently, have no array literals"),
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(op) => panic!("currently, have no builting identifiers that have array type, got {}", op),
            Identifier::Name(name) => {
                let (ty, identifier) = gen_variable(expr.pos(), name, ty);
                (ty, CudaExpression::Identifier(identifier))
            }
        }
    }
}

/// Generates a list of cuda identifiers that contain the data of the given variable with given type. For single-var-types,
/// only one result item will be yielded. For each result identifier, also the cuda type of this identifier is yielded.
///
/// The cuda type is usable for e.g. local variables, so primitive types have a non-pointer type.
///
/// In other words: this function generates the variables containing the data for var when given the variable var with type ty.

pub fn gen_variables<'a>(
    pos: &'a TextPosition,
    var: &'a Name,
    ty: &'a Type,
) -> impl 'a + Iterator<Item = (CudaType, CudaIdentifier)> {

    let value = gen_variable(pos, var, ty);

    let size_data = if is_mul_var_type(ty) { Some(()) } else { None };

    std::iter::once(value).chain(
        size_data
            .into_iter()
            .flat_map(move |_| gen_array_size(pos, var, ty)),
    )
}

/// Same as gen_variables, except that this function yields only one identifier and throws on multi-var-types

pub fn one_variable<I, R>(mut iter: I) -> (CudaType, R)
where
    I: Iterator<Item = (CudaType, R)>,
{

    let result = iter.next().unwrap();

    debug_assert!(iter.next().is_none());

    return result;
}

/// Generates a list of cuda expressions that evaluate to the data of the given variable. For single-var-types,
/// only one result item will be yielded. For each result identifier, also the cuda type of this identifier is yielded.
///
/// The difference to gen_variables() is that here, the expressions evaluate to the variables describing &var, i.e.
/// performing a type conversion. In particular, the values of the expressions can be used to change the original variable,
/// and the corresponding types are pointers. As a result, this behaves exactly as gen_variables for array types and for
/// & types.
///
/// Array entries can be modified through the result of this function, but array sizes cannot. For real output parameters,
/// consider gen_variables_as_output_params() instead.

pub fn gen_variables_for_view<'a>(
    pos: &'a TextPosition,
    var: &'a Name,
    ty: &'a Type,
) -> impl 'a + Iterator<Item = (CudaType, CudaExpression)> {

    let (mut cuda_type, cuda_val) = gen_variable(pos, var, ty);

    let expr = if !cuda_type.ptr && !cuda_type.owned {
        cuda_type.ptr = true;
        CudaExpression::AddressOf(Box::new(CudaExpression::Identifier(cuda_val)))
    } else if cuda_type.owned {
        cuda_type.ptr = true;
        cuda_type.owned = false;
        CudaExpression::Identifier(cuda_val)
    } else {

        CudaExpression::Identifier(cuda_val)
    };

    let size_data = if is_mul_var_type(ty) { Some(()) } else { None };

    std::iter::once((cuda_type, expr)).chain(size_data.into_iter().flat_map(move |_| {

        gen_array_size(pos, var, ty)
            .map(|(cuda_type, cuda_var)| (cuda_type, CudaExpression::Identifier(cuda_var)))
    }))
}

/// Generates a list of cuda expressions that evaluate to the data of the given variable. For single-var-types,
/// only one result item will be yielded. For each result identifier, also the cuda type of this identifier is yielded.
///
/// The difference to gen_variables() is that here, the expressions evaluate to the variables describing the non-view type
/// of var, i.e. performing a type conversion. In particular, if the original variable is scalar, the values of the expressions
/// cannot be used to change the it, and the corresponding types are non-pointer types. Arrays are not modified, as arrays
/// are always considered to be handled with pointers.
/// As a result, this behaves exactly as gen_variables for array types and for non-& types.
///
/// Array entries can be modified through the result of this function, but array sizes cannot. For real output parameters,
/// consider gen_variables_as_output_params() instead.

pub fn gen_variables_for_value<'a>(
    pos: &'a TextPosition,
    var: &'a Name,
    ty: &'a Type,
) -> impl 'a + Iterator<Item = (CudaType, CudaExpression)> {

    let (mut cuda_type, cuda_val) = gen_variable(pos, var, ty);

    let expr = if !is_mul_var_type(ty) && cuda_type.ptr {

        cuda_type.ptr = false;

        CudaExpression::deref(CudaExpression::Identifier(cuda_val))
    } else {

        CudaExpression::Identifier(cuda_val)
    };

    let size_data = if is_mul_var_type(ty) { Some(()) } else { None };

    std::iter::once((cuda_type, expr)).chain(size_data.into_iter().flat_map(move |_| {

        gen_array_size(pos, var, ty)
            .map(|(cuda_type, cuda_var)| (cuda_type, CudaExpression::Identifier(cuda_var)))
    }))
}

/// Generates a list of cuda expressions that evaluate to pointers on the data of the given variable. For single-var-types,
/// only one result item will be yielded. For each result identifier, also the cuda type of this identifier is yielded.
///
/// The difference to gen_variables() is that this function yields expressions that can be used to modify and/or replace the
/// value of the original variable.

pub fn gen_variables_for_output_params<'a>(
    pos: &'a TextPosition,
    var: &'a Name,
    ty: &'a Type,
) -> impl 'a + Iterator<Item = (CudaType, CudaExpression)> {

    gen_variables(pos, var, ty).map(|(ty, var)| {

        (
            CudaType {
                base: ty.base,
                constant: false,
                ptr: true,
                owned: ty.owned
            },
            CudaExpression::AddressOf(Box::new(CudaExpression::Identifier(var))),
        )
    })
}

pub fn gen_output_parameter_declaration<'a>(
    _pos: &'a TextPosition,
    arr: &'a ArrayType,
) -> impl 'a + Iterator<Item = (CudaType, CudaIdentifier)> {

    std::iter::once((
        CudaType {
            base: gen_primitive_type(arr.base),
            ptr: true,
            owned: true,
            constant: false
        },
        CudaIdentifier::OutputValueVar,
    ))
    .chain((0..arr.dimension).map(move |d| {

        (
            CudaType {
                base: CudaPrimitiveType::Index,
                ptr: true,
                owned: false,
                constant: false
            },
            CudaIdentifier::OutputArraySizeVar(d as u32),
        )
    }))
}

fn gen_defined_function_call<'stack, 'ast: 'stack, I>(
    call: &FunctionCall,
    function_name: &Name,
    function_type: &FunctionType,
    mut output_params: I,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<CudaExpression, OutputError>
where
    I: Iterator<Item = CudaExpression>,
{

    debug_assert!(call.function == Identifier::Name(function_name.clone()));

    let ast_lifetime = context.ast_lifetime();

    if !is_generated_with_output_parameter(function_type.return_type(ast_lifetime)) {
        assert!(output_params.next().is_none());
    }

    let params = call
        .parameters
        .iter()
        .zip(function_type.param_types(ast_lifetime))
        .flat_map(|(param, formal_param_ref)| {
            let formal_param = &*formal_param_ref;
            match param {
                Expression::Call(_) => {

                    assert!(!is_generated_with_output_parameter(Some(formal_param)));

                    if formal_param.is_view() {

                        Box::new(std::iter::once(gen_expression_for_view(param, context)))
                            as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>
                    } else {

                        Box::new(std::iter::once(gen_expression_for_value(param, context)))
                            as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>
                    }
                }
                Expression::Literal(lit) => Box::new(std::iter::once(Ok(CudaExpression::IntLiteral(
                    lit.value as i64,
                ))))
                    as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>,
                Expression::Variable(var) => match &var.identifier {
                    Identifier::Name(name) => {
                        if formal_param.is_view() {
                            Box::new(
                                gen_variables_for_view(var.pos(), name, &*formal_param)
                                    .map(|(_ty, p)| Ok(p)).collect::<Vec<_>>().into_iter())
                                as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>
                        } else {
                            Box::new(
                                gen_variables_for_value(var.pos(), name, &*formal_param)
                                    .map(|(_ty, p)| Ok(p)).collect::<Vec<_>>().into_iter())
                                as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>
                        }
                    }
                    Identifier::BuiltIn(_) => unimplemented!(),
                },
            }
        })
        .chain(output_params.map(|expr| Ok(expr)));

    Ok(CudaExpression::Call(
        CudaIdentifier::ValueVar(function_name.clone()),
        params.collect::<Result<Vec<CudaExpression>, OutputError>>()?,
    ))
}

fn gen_index_expression<'a, 'stack, 'ast: 'stack, I>(
    array: &Name,
    dim_count: u32,
    indices: I,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> impl Iterator<Item = Result<CudaExpression, OutputError>>
where
    I: Iterator<Item = &'a Expression>,
{

    let index = indices
        .enumerate()
        .map(|(dim, index)| {
            if dim as u32 + 1 == dim_count {

                Ok((AddSub::Plus, gen_expression_for_value(index, context)?))
            } else {

                Ok((
                    AddSub::Plus,
                    CudaExpression::Product(vec![
                        (MulDiv::Multiply, gen_expression_for_value(index, context)?),
                        (
                            MulDiv::Multiply,
                            CudaExpression::Identifier(CudaIdentifier::ArraySizeVar(
                                array.clone(),
                                dim as u32 + 1,
                            )),
                        ),
                    ]),
                ))
            }
        })
        .collect::<Result<Vec<(AddSub, CudaExpression)>, OutputError>>();

    std::iter::once(index.map(|i| {

        CudaExpression::AddressOf(Box::new(CudaExpression::Index(
            CudaIdentifier::ValueVar(array.clone()),
            Box::new(CudaExpression::Sum(i)),
        )))
    }))
}

fn gen_builtin_function_call<'stack, 'ast: 'stack>(
    call: &FunctionCall,
    func: BuiltInIdentifier,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<CudaExpression, OutputError> {

    fn process_sum_prod<'stack, 'ast: 'stack, E>(
        call: &FunctionCall,
        toggle_unary_func: BuiltInIdentifier,
        default_e: &E,
        toggled_e: &E,
        context: &mut dyn CudaContext<'stack, 'ast>,
    ) -> Result<Vec<(E, CudaExpression)>, OutputError>
    where
        E: Clone,
    {

        call.parameters
            .iter()
            .map(|p| match p {
                Expression::Call(subcall) if subcall.function == toggle_unary_func => Ok((
                    toggled_e.clone(),
                    gen_expression_for_value(&subcall.parameters[0], context)?,
                )),
                expr => Ok((default_e.clone(), gen_expression_for_value(expr, context)?)),
            })
            .collect::<Result<Vec<(E, CudaExpression)>, OutputError>>()
    }

    let fst = call.parameters.get(0);

    let snd = call.parameters.get(1);

    let mut cmp = |op: Cmp| -> Result<CudaExpression, OutputError> {

        Ok(CudaExpression::Comparison(
            op,
            Box::new(gen_expression_for_value(fst.unwrap(), context)?),
            Box::new(gen_expression_for_value(snd.unwrap(), context)?),
        ))
    };

    Ok(match func {
        BuiltInIdentifier::FunctionAdd => CudaExpression::Sum(process_sum_prod(call, BuiltInIdentifier::FunctionUnaryNeg, &AddSub::Plus, &AddSub::Minus, context)?),
        BuiltInIdentifier::FunctionMul => CudaExpression::Product(process_sum_prod(call, BuiltInIdentifier::FunctionUnaryDiv, &MulDiv::Multiply, &MulDiv::Divide, context)?),
        BuiltInIdentifier::FunctionAnd => CudaExpression::Conjunction(call.parameters.iter().map(|c| gen_expression_for_value(c, context)).collect::<Result<Vec<CudaExpression>, OutputError>>()?),
        BuiltInIdentifier::FunctionOr => CudaExpression::Disjunction(call.parameters.iter().map(|c| gen_expression_for_value(c, context)).collect::<Result<Vec<CudaExpression>, OutputError>>()?),
        BuiltInIdentifier::FunctionEq => cmp(Cmp::Eq)?,
        BuiltInIdentifier::FunctionNeq => cmp(Cmp::Neq)?,
        BuiltInIdentifier::FunctionLs => cmp(Cmp::Ls)?,
        BuiltInIdentifier::FunctionGt => cmp(Cmp::Gt)?,
        BuiltInIdentifier::FunctionLeq => cmp(Cmp::Leq)?,
        BuiltInIdentifier::FunctionGeq => cmp(Cmp::Geq)?,
        BuiltInIdentifier::FunctionUnaryDiv => CudaExpression::Product(vec![(MulDiv::Divide, gen_expression_for_value(fst.unwrap(), context)?)]),
        BuiltInIdentifier::FunctionUnaryNeg => CudaExpression::Sum(vec![(AddSub::Minus, gen_expression_for_value(fst.unwrap(), context)?)]),
        BuiltInIdentifier::FunctionIndex => match fst.unwrap() {
            Expression::Call(_) => panic!("Cuda backend currently cannot handle indexing into rvalues, should be extracted earlier"),
            Expression::Literal(_) => unimplemented!(),
            Expression::Variable(var) => match &var.identifier {
                Identifier::BuiltIn(op) => error_not_indexable_buildin_identifier(call.pos(), op).throw(),
                Identifier::Name(name) => gen_index_expression(name, call.parameters.len() as u32 - 1, call.parameters.iter().skip(1), context).next().unwrap()?
            }
        },
        _ => unimplemented!()
    })
}

/// Generates a cuda function call expression matching the given function call, with the output_params optionally passed as output parameters

pub fn gen_function_call<'stack, 'ast: 'stack, I>(
    call: &FunctionCall,
    mut output_params: I,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<CudaExpression, OutputError>
where
    I: Iterator<Item = CudaExpression>,
{

    match &call.function {
        Expression::Call(_) => Err(error_call_dynamic_expression(call.pos())),
        Expression::Literal(_) => {
            unimplemented!()
        }
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(op) => {

                assert!(output_params.next().is_none());

                gen_builtin_function_call(call, *op, context)
            }
            Identifier::Name(name) => {

                let function = context.get_scopes().get_defined(name, call.pos()).unwrap();

                gen_defined_function_call(call, name, &context.ast_lifetime().cast(function.get_type()).borrow().expect_callable(call.pos()).internal_error(), output_params, context)
            }
        },
    }
}

/// Generates a cuda expression matching the given expression. Requires the result of the expression to be of a single-var-type.

pub fn gen_expression<'stack, 'ast: 'stack>(
    expr: &Expression,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<CudaExpression, OutputError> {

    match expr {
        Expression::Call(call) => gen_function_call(call, std::iter::empty(), context),
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(_) => unimplemented!(),
            Identifier::Name(name) => {

                let expr_type = context.calculate_var_type(name, expr.pos());

                let result = one_variable(gen_variables(expr.pos(), name, &expr_type));

                Ok(CudaExpression::Identifier(result.1))
            }
        },
        Expression::Literal(lit) => Ok(CudaExpression::IntLiteral(lit.value as i64)),
    }
}

/// Generates a cuda expression matching the given expression. Requires the result of the expression to be of a single-var-type.
///
/// The difference to gen_expression() is that this function performs the type conversion to a view on the result type. For details,
/// see the difference between gen_variables() and gen_variables_for_view().

pub fn gen_expression_for_view<'stack, 'ast: 'stack>(
    expr: &Expression,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<CudaExpression, OutputError> {

    match expr {
        Expression::Call(call) => {
            if context.calculate_type(expr).is_view() {
                gen_function_call(call, std::iter::empty(), context)
            } else {
                Ok(CudaExpression::AddressOf(Box::new(gen_function_call(
                    call,
                    std::iter::empty(),
                    context,
                )?)))
            }
        }
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(_) => unimplemented!(),
            Identifier::Name(name) => {

                let expr_type = context.calculate_var_type(name, expr.pos());

                let result = one_variable(gen_variables_for_view(expr.pos(), name, &expr_type));

                Ok(result.1)
            }
        },
        Expression::Literal(lit) => Ok(CudaExpression::IntLiteral(lit.value as i64)),
    }
}

/// Generates a cuda expression matching the given expression. Requires the result of the expression to be of a single-var-type.
///
/// The difference to gen_expression() is that this function performs the type conversion to the result type without views. For details,
/// see the difference between gen_variables() and gen_variables_for_value().

pub fn gen_expression_for_value<'stack, 'ast: 'stack>(
    expr: &Expression,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<CudaExpression, OutputError> {

    match expr {
        Expression::Call(call) => {
            if context.calculate_type(expr).is_view() {

                Ok(CudaExpression::deref(gen_function_call(
                    call,
                    std::iter::empty(),
                    context,
                )?))
            } else {

                gen_function_call(call, std::iter::empty(), context)
            }
        }
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(_) => unimplemented!(),
            Identifier::Name(name) => {

                let expr_type = context.calculate_var_type(name, expr.pos());

                let result = one_variable(gen_variables_for_value(expr.pos(), name, &expr_type));

                Ok(result.1)
            }
        },
        Expression::Literal(lit) => Ok(CudaExpression::IntLiteral(lit.value as i64)),
    }
}

fn gen_array_copy_assignment<'stack, 'ast: 'stack>(
    assignee: &Name,
    assignee_type: &ArrayType,
    value: CudaExpression,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    let destination = CudaExpression::Identifier(CudaIdentifier::ValueVar(assignee.clone()));

    Ok(Box::new(CudaMemcpy {
        destination: destination,
        source: value,
        device: context.is_device_context(),
        length: CudaExpression::Identifier(CudaIdentifier::ArraySizeVar(assignee.clone(), 0)),
        base_type: gen_primitive_type(assignee_type.base)
    }))
}

fn gen_array_copy_assignment_size_assertion<'stack, 'ast: 'stack, I>(
    assignee: &Name,
    assignee_type: &ArrayType,
    value_sizes: I,
    _context: &mut dyn CudaContext<'stack, 'ast>,
) -> impl Iterator<Item = Result<Box<dyn CudaStatement>, OutputError>>
where
    I: Iterator<Item = CudaExpression>,
{
    let assignee_copy = assignee.clone();

    (0..assignee_type.dimension).zip(value_sizes).map(move |(d, value_size)| {

        let assignee_size = CudaIdentifier::ArraySizeVar(assignee_copy.clone(), d as u32);

        Ok(Box::new(CudaAssert {
            expr: CudaExpression::Comparison(
                Cmp::Eq,
                Box::new(CudaExpression::Identifier(assignee_size)),
                Box::new(value_size),
            ),
        }) as Box<dyn CudaStatement>)
    })
}

fn gen_array_move_assignment_from_call<'stack, 'ast: 'stack>(
    pos: &TextPosition,
    assignee: &Name,
    value: &FunctionCall,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    let ty = context.calculate_var_type(assignee, pos).clone();

    Ok(Box::new(gen_function_call(
        value,
        gen_variables_for_output_params(pos, assignee, &ty).map(|(_, v)| v),
        context,
    )?))
}

fn gen_array_move_assignment_from_var<'stack, 'ast: 'stack>(
    pos: &TextPosition,
    assignee: &Name,
    value: &Name,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> impl Iterator<Item = Result<Box<dyn CudaStatement>, OutputError>> {

    let var_ty = context.calculate_var_type(assignee, pos);
    let ty = var_ty.expect_array(pos).internal_error();

    let assignee_copy = assignee.clone();

    let value_copy = value.clone();

    std::iter::once(CudaAssignment {
        assignee: CudaExpression::Identifier(CudaIdentifier::ValueVar(assignee.clone())),
        value: CudaExpression::Move(Box::new(CudaExpression::Identifier(CudaIdentifier::ValueVar(value.clone())))),
    })
    .chain((0..ty.dimension).map(move |d| CudaAssignment {
        assignee: CudaExpression::Identifier(CudaIdentifier::ArraySizeVar(
            assignee_copy.clone(),
            d as u32,
        )),
        value: CudaExpression::Identifier(CudaIdentifier::ArraySizeVar(value_copy.clone(), d as u32)),
    }))
    .map(|a| Ok(Box::new(a) as Box<dyn CudaStatement>))
}

pub fn gen_scalar_assignment<'stack, 'ast: 'stack>(
    assignee: &Expression,
    value: &Expression,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<CudaAssignment, OutputError> {

    Ok(CudaAssignment {
        assignee: gen_expression_for_value(assignee, context)?,
        value: gen_expression_for_value(value, context)?,
    })
}

pub fn gen_call_array_result_in_tmp_var<'stack, 'ast: 'stack>(
    _pos: &TextPosition,
    result_type: &ArrayType,
    call: &FunctionCall,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> impl Iterator<Item = Result<Box<dyn CudaStatement>, OutputError>> {

    let declarations = std::iter::once(CudaVarDeclaration {
        var: CudaIdentifier::TmpVar,
        value: Some(CudaExpression::Nullptr),
        var_type: CudaType {
            base: gen_primitive_type(result_type.base),
            owned: true,
            ptr: false,
            constant: false
        },
    })
    .chain((0..result_type.dimension).map(|d| CudaVarDeclaration {
        var: CudaIdentifier::TmpSizeVar(d as u32),
        value: Some(CudaExpression::IntLiteral(0)),
        var_type: CudaType {
            base: CudaPrimitiveType::Index,
            owned: false,
            ptr: false,
            constant: false
        },
    }))
    .map(|d| Box::new(d) as Box<dyn CudaStatement>)
    .map(|d| Ok(d));

    let output_params = std::iter::once(CudaExpression::AddressOf(Box::new(
        CudaExpression::Identifier(CudaIdentifier::TmpVar),
    )))
    .chain((0..result_type.dimension).map(|d| {

        CudaExpression::AddressOf(Box::new(CudaExpression::Identifier(
            CudaIdentifier::TmpSizeVar(d as u32),
        )))
    }));

    declarations.chain(std::iter::once(
        gen_function_call(call, output_params, context)
            .map(|d| Box::new(d) as Box<dyn CudaStatement>),
    ))
}

pub fn gen_array_checked_copy_assignment_from_var<'stack, 'ast: 'stack>(
    pos: &TextPosition,
    assignee: &Name,
    assignee_type: &Type,
    value: &Name,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    let array_type = match assignee_type {
        Type::Function(_) => unimplemented!(),
        Type::Array(arr) => arr,
        Type::View(view) => &view.base,
        Type::TestType => error_test_type(pos),
        Type::JumpLabel => error_jump_label_var_type(pos).throw()
    };

    let size_check = gen_array_copy_assignment_size_assertion(
        assignee,
        array_type,
        (0..array_type.dimension)
            .map(|d| CudaIdentifier::ArraySizeVar(value.clone(), d as u32))
            .map(CudaExpression::Identifier),
        context,
    );

    let copy = gen_array_copy_assignment(
        assignee,
        array_type,
        CudaExpression::Identifier(CudaIdentifier::ValueVar(value.clone())),
        context,
    );

    Ok(Box::new(CudaBlock {
        statements: size_check
            .chain(std::iter::once(copy))
            .collect::<Result<Vec<Box<dyn CudaStatement>>, OutputError>>()?,
    }))
}

pub fn gen_array_assignment<'stack, 'ast: 'stack>(
    pos: &TextPosition,
    assignee: &Name,
    assignee_type: &Type,
    value: &Expression,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    let without_view_type = assignee_type.clone().without_view();
    let array_type = without_view_type.expect_array(pos).internal_error();

    match assignee_type {
        Type::TestType => error_test_type(pos),
        Type::JumpLabel => error_jump_label_var_type(pos).throw(),
        Type::Function(_) => unimplemented!(),
        Type::Array(_) => match value {
            Expression::Call(call) => {
                gen_array_move_assignment_from_call(pos, assignee, call, context)
            }
            Expression::Variable(var) => match &var.identifier {
                Identifier::BuiltIn(_) => unimplemented!(),
                Identifier::Name(val_name)
                    if context.calculate_var_type(val_name, pos).is_view() =>
                {
                    gen_array_checked_copy_assignment_from_var(
                        pos,
                        assignee,
                        assignee_type,
                        val_name,
                        context,
                    )
                }
                Identifier::Name(val_name) => {

                    let assignments =
                        gen_array_move_assignment_from_var(pos, assignee, val_name, context)
                            .collect::<Result<Vec<_>, _>>()?;

                    Ok(Box::new(CudaBlock {
                        statements: assignments,
                    }))
                }
            },
            Expression::Literal(_) => unimplemented!(),
        },
        Type::View(view) => match value {
            Expression::Call(call) => {

                let tmp_var_init =
                    gen_call_array_result_in_tmp_var(pos, &array_type, call, context);

                let size_check = gen_array_copy_assignment_size_assertion(
                    assignee,
                    &view.base,
                    (0..array_type.dimension)
                        .map(|d| CudaIdentifier::TmpSizeVar(d as u32))
                        .map(CudaExpression::Identifier),
                    context,
                );

                let copy = gen_array_copy_assignment(
                    assignee,
                    &view.base,
                    CudaExpression::Identifier(CudaIdentifier::TmpVar),
                    context,
                );

                Ok(Box::new(CudaBlock {
                    statements: tmp_var_init
                        .chain(size_check)
                        .chain(std::iter::once(copy))
                        .collect::<Result<Vec<Box<dyn CudaStatement>>, OutputError>>()?,
                }))
            }
            Expression::Variable(var) => match &var.identifier {
                Identifier::BuiltIn(_) => unimplemented!(),
                Identifier::Name(val_name) => gen_array_checked_copy_assignment_from_var(
                    pos,
                    assignee,
                    assignee_type,
                    val_name,
                    context,
                ),
            },
            Expression::Literal(_) => unimplemented!(),
        }
    }
}

pub fn gen_assignment<'stack, 'ast: 'stack>(
    statement: &Assignment,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    let result: Result<Box<dyn CudaStatement>, OutputError> = match &statement.assignee {
        Expression::Call(call) => {

            if call.function != BuiltInIdentifier::FunctionIndex {

                error_rvalue_not_assignable(statement.pos()).throw()
            }

            Ok(Box::new(gen_scalar_assignment(
                &statement.assignee,
                &statement.value,
                context,
            )?))
        }
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(_) => error_rvalue_not_assignable(statement.pos()).throw(),
            Identifier::Name(name) => {

                let ty = context.calculate_var_type(name, statement.pos()).clone();

                if is_mul_var_type(&ty) {
                    gen_array_assignment(statement.pos(), name, &ty, &statement.value, context)
                } else {
                    Ok(Box::new(gen_scalar_assignment(
                        &statement.assignee,
                        &statement.value,
                        context,
                    )?))
                }
            }
        },
        Expression::Literal(_) => error_rvalue_not_assignable(statement.pos()).throw(),
    };
    return result;
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::Parser;
#[cfg(test)]
use super::kernel_data::collect_functions_global;
#[cfg(test)]
use super::context::CudaContextImpl;
#[cfg(test)]
use super::super::analysis::defs_test::EnvironmentBuilder;
#[cfg(test)]
use super::context_test::*;

#[test]
fn test_gen_expression() {
    let mut environment = EnvironmentBuilder::new()
        .add_array_def("a", PrimitiveType::Int, 0)
        .add_array_def("b", PrimitiveType::Int, 0)
        .add_array_def("c", PrimitiveType::Int, 0)
        .add_array_def("d", PrimitiveType::Int, 1)
        .add_array_def("e", PrimitiveType::Int, 0);

    let expr = Expression::parse(&mut fragment_lex("(a + b * c) / d[e,]"), environment.types()).unwrap();

    let (program, defs) = mock_program(environment);
    let (functions, kernels) = collect_functions_global(&program).unwrap();
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::new(&program, &functions, &kernels));
    context.enter_scope(&defs);

    assert_eq!("(a_ + b_ * c_) / d_[e_]", output(&gen_expression(&expr, &mut *context).unwrap()));
}

#[test]

fn test_gen_expression_function_call() {
    let mut environment = EnvironmentBuilder::new()
        .add_func_def("a").add_view_param(PrimitiveType::Int, 1).return_type(PrimitiveType::Int, 1)
        .add_array_def("foo", PrimitiveType::Int, 2)
        .add_array_def("b", PrimitiveType::Int, 0)
        .add_array_def("c", PrimitiveType::Int, 0);

    let expr = Expression::parse(&mut fragment_lex("foo[a(b,), c,]"), environment.types()).unwrap();

    let (program, defs) = mock_program(environment);
    let (functions, kernels) = collect_functions_global(&program).unwrap();
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::new(&program, &functions, &kernels));
    context.enter_scope(&defs);

    assert_eq!("&foo_[a_(b_, b_d0) * foo_d1 + c_]", output(&gen_expression(&expr, &mut *context).unwrap()));
}

#[test]

fn test_gen_expression_pass_index_expression_by_view() {
    let mut environment = EnvironmentBuilder::new()
        .add_func_def("foo").add_view_param(PrimitiveType::Int, 0).return_type(PrimitiveType::Int, 0)
        .add_array_def("a", PrimitiveType::Int, 0);

    let expr = Expression::parse(&mut fragment_lex("foo(a[0,],)"), environment.types()).unwrap();

    let (program, defs) = mock_program(environment);
    let (functions, kernels) = collect_functions_global(&program).unwrap();
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::new(&program, &functions, &kernels));
    context.enter_scope(&defs);

    assert_eq!("foo_(&a_[0])", output(&gen_expression(&expr, &mut *context).unwrap()));
}

#[test]

fn test_gen_expression_pass_index_expression_by_value() {
    let mut environment = EnvironmentBuilder::new()
        .add_func_def("foo").add_array_param(PrimitiveType::Int, 0).return_type(PrimitiveType::Int, 0)
        .add_array_def("a", PrimitiveType::Int, 0);

    let expr = Expression::parse(&mut fragment_lex("foo(a[0,],)"), environment.types()).unwrap();

    let (program, defs) = mock_program(environment);
    let (functions, kernels) = collect_functions_global(&program).unwrap();
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::new(&program, &functions, &kernels));
    context.enter_scope(&defs);

    assert_eq!("foo_(a_[0])", output(&gen_expression(&expr, &mut *context).unwrap()));
}

#[test]
fn test_gen_assignment_var_to_var() {
    let mut environment = EnvironmentBuilder::new()
        .add_array_def("a", PrimitiveType::Int, 2)
        .add_array_def("b", PrimitiveType::Int, 2);

    let assignment = Statement::parse(&mut fragment_lex("a = b;"), environment.types())
        .unwrap()
        .any_box()
        .downcast::<Assignment>()
        .unwrap();

    let (program, defs) = mock_program(environment);
    let (functions, kernels) = collect_functions_global(&program).unwrap();
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::new(&program, &functions, &kernels));
    context.enter_scope(&defs);

    assert_eq!(
        "{
    a_ = std::move(b_);
    a_d0 = b_d0;
    a_d1 = b_d1;
}", output(&*gen_assignment(&*assignment, &mut *context).unwrap()));
}

#[test]
fn test_gen_assignment_array_view_to_array() {
    let mut environment = EnvironmentBuilder::new()
        .add_array_def("a", PrimitiveType::Int, 2)
        .add_view_def("b", PrimitiveType::Int, 2);

    let assignment = Statement::parse(&mut fragment_lex("a = b;"), environment.types())
        .unwrap()
        .any_box()
        .downcast::<Assignment>()
        .unwrap();

    let (program, defs) = mock_program(environment);
    let (functions, kernels) = collect_functions_global(&program).unwrap();
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::new(&program, &functions, &kernels));
    context.enter_scope(&defs);

    assert_eq!(
        "{
    assert(a_d0 == b_d0);
    assert(a_d1 == b_d1);
    checkCudaStatus(cudaMemcpy(a_, b_, sizeof(int) * a_d0, cudaMemcpyDeviceToDevice));
}", output(&*gen_assignment(&*assignment, &mut *context).unwrap()));
}

#[test]

fn test_gen_assignment_call_result_to_array_view() {
    let mut environment = EnvironmentBuilder::new()
        .add_func_def("foo").add_view_param(PrimitiveType::Int, 1).add_array_param(PrimitiveType::Int,0).return_type(PrimitiveType::Int, 2)
        .add_view_def("b", PrimitiveType::Int, 2)
        .add_array_def("a", PrimitiveType::Int, 1)
        .add_view_def("c", PrimitiveType::Int, 0);

    let assignment = Statement::parse(&mut fragment_lex("b = foo(a, c,);"), environment.types())
        .unwrap()
        .any_box()
        .downcast::<Assignment>()
        .unwrap();

    let (program, defs) = mock_program(environment);
    let (functions, kernels) = collect_functions_global(&program).unwrap();
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::new(&program, &functions, &kernels));
    context.enter_scope(&defs);

    assert_eq!(
        "{
    DevPtr<int> tmp = nullptr;
    unsigned int tmpd0 = 0;
    unsigned int tmpd1 = 0;
    foo_(a_, a_d0, c_, &tmp, &tmpd0, &tmpd1);
    assert(b_d0 == tmpd0);
    assert(b_d1 == tmpd1);
    checkCudaStatus(cudaMemcpy(b_, tmp, sizeof(int) * b_d0, cudaMemcpyDeviceToDevice));
}", output(&*gen_assignment(&*assignment, &mut *context).unwrap()));
}
