use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::super::analysis::type_error::*;
use super::context::CudaContext;
use super::ast::*;

fn error_call_dynamic_expression(pos: &TextPosition) -> OutputError {
    OutputError::UnsupportedCode(pos.clone(), format!("Calling dynamic expressions is not supported by the Cuda Backend"))
}

pub fn is_mul_var_type(ty: &Type) -> bool {
    match ty {
        Type::Array(_, dim) => *dim > 0,
        Type::View(viewn) => is_mul_var_type(viewn),
        _ => false
    }
}

pub fn is_generated_with_output_parameter(return_type: Option<&Box<Type>>) -> bool {
    return_type.map(|ty| &**ty).map(is_mul_var_type).unwrap_or(false)
}

pub fn gen_primitive_ptr_type(value: &PrimitiveType, ptr_count: u32) -> CudaType {
    assert_eq!(*value, PrimitiveType::Int);
    CudaType {
        base: CudaPrimitiveType::Int,
        constant: false,
        ptr_count: ptr_count
    }
}

pub fn expect_array_type(value: &Type) -> (PrimitiveType, u32) {
    match value {
        Type::Array(base, dim) => (*base, *dim),
        Type::View(viewn) => match &**viewn {
            Type::Array(base, dim) => (*base, *dim),
            _ => panic!("expect_array_type() called for non-array type {:?}", value)
        },
        _ => panic!("expect_array_type() called for non-array type {:?}", value)
    }
}

pub fn gen_array_size_vars<'a>(pos: &TextPosition, name: &'a Name, ty: &Type) -> impl 'a + Iterator<Item = (CudaType, CudaIdentifier)> {
    let (_, dim_count) = expect_array_type(ty);
    (0..dim_count).map(move |d| (CudaType {
        ptr_count: 0,
        constant: false,
        base: CudaPrimitiveType::Index
    }, CudaIdentifier::ArraySizeVar(name.clone(), d)))
}

pub fn gen_array_size_exprs<'a>(expr: &'a Expression, ty: &Type) -> impl 'a + Iterator<Item = (CudaType, CudaExpression)> {
    match expr {
        Expression::Call(call) => panic!("gen_array_size_vars() currently accepts no call expressions, as builtin calls do not yield arrays, and user-defined calls should be extracted"),
        Expression::Literal(lit) => panic!("currently, have no array literals"),
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(op) => panic!("currently, have no builting identifiers that have array type, got {}", op),
            Identifier::Name(name) => gen_array_size_vars(expr.pos(), name, ty).map(|(ty, var)| (ty, CudaExpression::Identifier(var)))
        }
    }
}

pub fn gen_value_var(pos: &TextPosition, name: &Name, ty: &Type) -> (CudaType, CudaIdentifier) {
    match ty {
        Type::Array(base, dim) => (gen_primitive_ptr_type(base, 1), CudaIdentifier::ValueVar(name.clone())),
        Type::Function(_, _) => panic!("Cannot represent a function in cuda variables"),
        Type::JumpLabel => panic!("Cannot represent a jump label in cuda variables"),
        Type::Primitive(base) => (gen_primitive_ptr_type(base, 0), CudaIdentifier::ValueVar(name.clone())),
        Type::TestType => error_test_type(pos),
        Type::View(viewn) => match &**viewn {
            Type::Array(base, dim) => (gen_primitive_ptr_type(base, 1), CudaIdentifier::ValueVar(name.clone())),
            Type::Function(_, _) => panic!("Cannot represent a function in cuda variables"),
            Type::JumpLabel => panic!("Cannot represent a jump label in cuda variables"),
            Type::Primitive(base) => (gen_primitive_ptr_type(base, 1), CudaIdentifier::ValueVar(name.clone())),
            Type::TestType => error_test_type(pos),
            Type::View(_) => error_nested_view(pos).throw()
        }
    }
}

/// Generates a list of cuda identifiers that contain the data of the given variable with given type. For single-var-types,
/// only one result item will be yielded. For each result identifier, also the cuda type of this identifier is yielded.
/// 
/// The cuda type is usable for e.g. local variables, so primitive types have a non-pointer type, and array size variables
/// cannot be changed
pub fn gen_variables<'a>(pos: &'a TextPosition, var: &'a Name, ty: &'a Type) -> impl 'a + Iterator<Item = (CudaType, CudaIdentifier)> {
    let value = gen_value_var(pos, var, ty);
    let size_data = if ty.is_array() { Some(()) } else { None };
    std::iter::once(value).chain(size_data.into_iter().flat_map(move |_| gen_array_size_vars(pos, var, ty)))
}

/// Same as gen_variables, except that this function yields only one identifier and throws on multi-var-types
pub fn gen_variable(pos: &TextPosition, var: &Name, ty: &Type) -> (CudaType, CudaIdentifier) {
    assert!(!is_mul_var_type(ty));
    let mut iter = gen_variables(pos, var, ty);
    let result = iter.next().unwrap();
    debug_assert!(iter.next().is_none());
    return result;
}

/// Generates a list of cuda expressions that evaluate to the data of the given variable. For single-var-types,
/// only one result item will be yielded. For each result identifier, also the cuda type of this identifier is yielded.
/// 
/// The difference to gen_variables() is that here, all expressions evaluate to values that can be used to modify the value of 
/// original variable, and the yielded types reflect that (i.e. they are pointers). Therefore, this result is usable 
/// e.g. for parameters that are passed by reference.
/// Array entries can be modified through the result of this function, but array sizes cannot. For real output parameters,
/// consider gen_variables_as_output_params() instead.
pub fn gen_variables_as_view<'a>(pos: &'a TextPosition, var: &'a Name, ty: &'a Type) -> impl 'a + Iterator<Item = (CudaType, CudaExpression)> {
    gen_variables(pos, var, ty).enumerate().map(|(index, (mut ty, var))| {
        // TODO: make this more beautiful: only the value (not the dimension vars) get promoted to view
        if index == 0 && ty.ptr_count == 0 {
            ty.ptr_count = 1;
            (ty, CudaExpression::AddressOf(Box::new(CudaExpression::Identifier(var))))
        } else if index == 0 {
            (ty, CudaExpression::Identifier(var))
        } else {
            debug_assert!(ty.ptr_count == 0);
            debug_assert!(ty.base == CudaPrimitiveType::Index);
            (CudaType {
                ptr_count: ty.ptr_count,
                constant: true,
                base: ty.base
            }, CudaExpression::Identifier(var))
        }
    })
}

/// Generates a list of cuda expressions that evaluate to pointers on the data of the given variable. For single-var-types,
/// only one result item will be yielded. For each result identifier, also the cuda type of this identifier is yielded.
/// 
/// The difference to gen_variables() is that this function yields expressions that can be used to modify and/or replace the
/// value of the original variable.
pub fn gen_variables_as_output_params<'a>(pos: &'a TextPosition, var: &'a Name, ty: &'a Type) -> impl 'a + Iterator<Item = (CudaType, CudaExpression)> {
    gen_variables(pos, var, ty).map(|(ty, var)| (CudaType {
        base: ty.base,
        constant: false,
        ptr_count: ty.ptr_count + 1
    }, CudaExpression::AddressOf(Box::new(CudaExpression::Identifier(var)))))
}

fn gen_defined_function_call<'stack, 'ast: 'stack, I>(call: &FunctionCall, function_name: &Name, function_type: &Type, mut output_params: I, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<CudaExpression, OutputError> 
    where I: Iterator<Item = CudaExpression>
{
    debug_assert!(call.function == Identifier::Name(function_name.clone()));

    let (param_types, return_type) = match function_type {
        Type::Function(param_types, return_type) => (param_types, return_type),
        _ => panic!("gen_defined_function_call() expected to get function type")
    };
    if !is_generated_with_output_parameter(return_type.as_ref()) {
        assert!(output_params.next().is_none());
    }
    let params = call.parameters.iter().zip(param_types.iter()).flat_map(|(param, formal_param)| match param {
        Expression::Call(subcall) => {
            assert!(!is_generated_with_output_parameter(Some(formal_param)));
            let result = gen_function_call(subcall, std::iter::empty(), context);
            if formal_param.is_view() {
                Box::new(std::iter::once(result.map(Box::new).map(CudaExpression::AddressOf))) as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>
            } else {
                Box::new(std::iter::once(result)) as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>
            }
        },
        Expression::Literal(_lit) => unimplemented!(),
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => if formal_param.is_view() {
                Box::new(gen_variables_as_view(var.pos(), name, &formal_param).map(|(_ty, p)| Ok(p))) as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>
            } else {
                Box::new(gen_variables(var.pos(), name, &formal_param).map(|(_ty, p)| Ok(CudaExpression::Identifier(p)))) as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>
            },
            Identifier::BuiltIn(_) => unimplemented!()
        }
    }).chain(output_params.map(|expr| Ok(expr)));
    Ok(CudaExpression::Call(CudaIdentifier::ValueVar(function_name.clone()), params.collect::<Result<Vec<CudaExpression>, OutputError>>()?))
}

fn gen_index_expression<'a, 'stack, 'ast: 'stack, I>(array: &Name, dim_count: u32, indices: I, context: &mut dyn CudaContext<'stack, 'ast>) -> impl Iterator<Item = Result<CudaExpression, OutputError>> 
    where I: Iterator<Item = &'a Expression>
{
    let index = indices.enumerate().map(|(dim, index)| if dim as u32 + 1 == dim_count {
        Ok((AddSub::Plus, gen_expression(index, context)?))
    } else {
        Ok((AddSub::Plus, CudaExpression::Product(vec![
            (MulDiv::Multiply, gen_expression(index, context)?),
            (MulDiv::Multiply, CudaExpression::Identifier(CudaIdentifier::ArraySizeVar(array.clone(), dim as u32 + 1)))
        ])))
    }).collect::<Result<Vec<(AddSub, CudaExpression)>, OutputError>>();

    std::iter::once(index.map(|i| CudaExpression::Index(CudaIdentifier::ValueVar(array.clone()), Box::new(CudaExpression::Sum(i)))))
}

fn gen_builtin_function_call<'stack, 'ast: 'stack>(call: &FunctionCall, func: BuiltInIdentifier, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<CudaExpression, OutputError> 
{
    fn process_row_op<'stack, 'ast: 'stack, E>(call: &FunctionCall, toggle_unary_func: BuiltInIdentifier, default_e: &E, toggled_e: &E, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Vec<(E, CudaExpression)>, OutputError> 
    where E: Clone
    {
        call.parameters.iter().map(|p| {
            match p {
                Expression::Call(subcall) if subcall.function == toggle_unary_func => Ok((toggled_e.clone(), gen_expression(&subcall.parameters[0], context)?)),
                expr => Ok((default_e.clone(), gen_expression(expr, context)?))
            }
        }).collect::<Result<Vec<(E, CudaExpression)>, OutputError>>()
    }
    let fst = call.parameters.get(0);
    let snd = call.parameters.get(1);
    
    let mut cmp = |op: Cmp| -> Result<CudaExpression, OutputError> {
        Ok(CudaExpression::Comparison(op, Box::new(gen_expression(fst.unwrap(), context)?), Box::new(gen_expression(snd.unwrap(), context)?)))
    };

    Ok(match func {
        BuiltInIdentifier::FunctionAdd => CudaExpression::Sum(process_row_op(call, BuiltInIdentifier::FunctionUnaryNeg, &AddSub::Plus, &AddSub::Minus, context)?),
        BuiltInIdentifier::FunctionMul => CudaExpression::Product(process_row_op(call, BuiltInIdentifier::FunctionUnaryDiv, &MulDiv::Multiply, &MulDiv::Divide, context)?),
        BuiltInIdentifier::FunctionAnd => CudaExpression::Conjunction(call.parameters.iter().map(|c| gen_expression(c, context)).collect::<Result<Vec<CudaExpression>, OutputError>>()?),
        BuiltInIdentifier::FunctionOr => CudaExpression::Disjunction(call.parameters.iter().map(|c| gen_expression(c, context)).collect::<Result<Vec<CudaExpression>, OutputError>>()?),
        BuiltInIdentifier::FunctionEq => cmp(Cmp::Eq)?,
        BuiltInIdentifier::FunctionNeq => cmp(Cmp::Neq)?,
        BuiltInIdentifier::FunctionLs => cmp(Cmp::Ls)?,
        BuiltInIdentifier::FunctionGt => cmp(Cmp::Gt)?,
        BuiltInIdentifier::FunctionLeq => cmp(Cmp::Leq)?,
        BuiltInIdentifier::FunctionGeq => cmp(Cmp::Geq)?,
        BuiltInIdentifier::FunctionUnaryDiv => CudaExpression::Product(vec![(MulDiv::Divide, gen_expression(fst.unwrap(), context)?)]),
        BuiltInIdentifier::FunctionUnaryNeg => CudaExpression::Sum(vec![(AddSub::Minus, gen_expression(fst.unwrap(), context)?)]),
        BuiltInIdentifier::FunctionIndex => match fst.unwrap() {
            Expression::Call(_) => panic!("Cuda backend currently cannot handle indexing into rvalues, should be extracted earlier"),
            Expression::Literal(_) => unimplemented!(),
            Expression::Variable(var) => match &var.identifier {
                Identifier::BuiltIn(op) => error_not_indexable_buildin_identifier(call.pos(), op).throw(),
                Identifier::Name(name) => gen_index_expression(name, call.parameters.len() as u32 - 1, call.parameters.iter().skip(1), context).next().unwrap()?
            }
        }
    })
}

/// Generates a cuda function call expression matching the given function call, with the output_params optionally passed as output parameters
pub fn gen_function_call<'stack, 'ast: 'stack, I>(call: &FunctionCall, mut output_params: I, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<CudaExpression, OutputError> 
    where I: Iterator<Item = CudaExpression>
{
    match &call.function {
        Expression::Call(_) => Err(error_call_dynamic_expression(call.pos())),
        Expression::Literal(_) => error_not_callable(call.pos(), &Type::Primitive(PrimitiveType::Int)).throw(),
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(op) => {
                assert!(output_params.next().is_none());
                gen_builtin_function_call(call, *op, context)
            },
            Identifier::Name(name) => {
                let function = context.get_scopes().get_defined(name, call.pos()).unwrap();
                gen_defined_function_call(call, name, &function.calc_type(), output_params, context)
            }
        }
    }
}

/// Generates a cuda expression matching the given expression. Requires the result of the expression to be of a single-var-type
pub fn gen_expression<'stack, 'ast: 'stack>(expr: &Expression, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<CudaExpression, OutputError>  {
    match expr {
        Expression::Call(call) => gen_function_call(call, std::iter::empty(), context),
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(_) => unimplemented!(),
            Identifier::Name(name) => {
                let expr_type = context.calculate_var_type(name, expr.pos());
                let mut result = gen_variables(expr.pos(), name, &expr_type);
                let r = result.next().unwrap();
                // if result contains more values, it is a multi-var-type
                assert!(result.next().is_none());
                Ok(CudaExpression::Identifier(r.1))
            }
        },
        Expression::Literal(lit) => Ok(CudaExpression::IntLiteral(lit.value as i64))
    }
}

fn gen_array_copy_assignment<'stack, 'ast: 'stack>(assignee: &Name, assignee_type: &Type, value: CudaExpression, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    let (base_type, _) = expect_array_type(&assignee_type.clone().without_view());
    let destination = CudaExpression::Identifier(CudaIdentifier::ValueVar(assignee.clone()));
    Ok(Box::new(CudaMemcpy {
        destination: destination,
        source: value,
        device: context.is_device_context(),
        length: CudaExpression::Identifier(CudaIdentifier::ArraySizeVar(assignee.clone(), 0)),
        base_type: gen_primitive_ptr_type(&base_type, 0)
    }))
}

fn gen_array_copy_assignment_size_assertion<'stack, 'ast: 'stack, I>(assignee: &Name, assignee_type: &Type, value_sizes: I, _context: &mut dyn CudaContext<'stack, 'ast>) -> impl Iterator<Item = Result<Box<dyn CudaStatement>, OutputError>> 
    where I: Iterator<Item = CudaExpression>
{
    let (_, dim) = expect_array_type(&assignee_type.clone().without_view());
    let assignee_copy = assignee.clone();
    (0..dim).zip(value_sizes).map(move |(d, value_size)| {
        let assignee_size = CudaIdentifier::ArraySizeVar(assignee_copy.clone(), d);
        Ok(Box::new(CudaAssert { expr: CudaExpression::Comparison(Cmp::Eq, Box::new(CudaExpression::Identifier(assignee_size)), Box::new(value_size)) }) as Box<dyn CudaStatement>)
    })
}

fn gen_array_move_assignment_from_call<'stack, 'ast: 'stack>(pos: &TextPosition, assignee: &Name, value: &FunctionCall, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    let ty = context.calculate_var_type(assignee, pos);
    Ok(Box::new(gen_function_call(value, gen_variables_as_output_params(pos, assignee, &ty).map(|(_, v)| v), context)?))
}

fn gen_array_move_assignment_from_var<'stack, 'ast: 'stack>(pos: &TextPosition, assignee: &Name, value: &Name, context: &mut dyn CudaContext<'stack, 'ast>) -> impl Iterator<Item = Result<Box<dyn CudaStatement>, OutputError>> {
    let ty = context.calculate_var_type(assignee, pos);
    let (_base_type, dim) = expect_array_type(&ty);
    let assignee_copy = assignee.clone();
    let value_copy = value.clone();

    std::iter::once(CudaAssignment {
        assignee: CudaExpression::Identifier(CudaIdentifier::ValueVar(assignee.clone())),
        value: CudaExpression::Identifier(CudaIdentifier::ValueVar(value.clone()))
    }).chain((0..dim).map(move |d| CudaAssignment {
        assignee: CudaExpression::Identifier(CudaIdentifier::ArraySizeVar(assignee_copy.clone(), d)),
        value: CudaExpression::Identifier(CudaIdentifier::ArraySizeVar(value_copy.clone(), d))
    })).map(|a| Ok(Box::new(a) as Box<dyn CudaStatement>))
}

pub fn gen_scalar_assignment<'stack, 'ast: 'stack>(assignee: &Expression, value: &Expression, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<CudaAssignment, OutputError> {
    Ok(CudaAssignment {
        assignee: gen_expression(assignee, context)?,
        value: gen_expression(value, context)?
    })
}

pub fn gen_call_array_result_in_tmp_var<'stack, 'ast: 'stack>(_pos: &TextPosition, result_type: &Type, call: &FunctionCall, context: &mut dyn CudaContext<'stack, 'ast>) -> impl Iterator<Item = Result<Box<dyn CudaStatement>, OutputError>> {
    let (base_type, dim) = expect_array_type(result_type);
    let declarations = std::iter::once(CudaVarDeclaration {
        var: CudaIdentifier::TmpVar,
        value: Some(CudaExpression::Nullptr),
        var_type: gen_primitive_ptr_type(&base_type, 1)
    }).chain((0..dim).map(|d| CudaVarDeclaration {
        var: CudaIdentifier::TmpSizeVar(d),
        value: Some(CudaExpression::IntLiteral(0)),
        var_type: CudaType {
            base: CudaPrimitiveType::Index,
            constant: false,
            ptr_count: 0
        }
    })).map(|d| Box::new(d) as Box<dyn CudaStatement>).map(|d| Ok(d));

    let output_params = std::iter::once(
        CudaExpression::AddressOf(Box::new(CudaExpression::Identifier(CudaIdentifier::TmpVar)))
    ).chain((0..dim).map(|d|
        CudaExpression::AddressOf(Box::new(CudaExpression::Identifier(CudaIdentifier::TmpSizeVar(d))))
    ));
    declarations.chain(std::iter::once(gen_function_call(call, output_params, context).map(|d| Box::new(d) as Box<dyn CudaStatement>)))
}

pub fn gen_array_checked_copy_assignment_from_var<'stack, 'ast: 'stack>(pos: &TextPosition, assignee: &Name, assignee_type: &Type, value: &Name, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    let array_type = assignee_type.clone().without_view();
    let (_, dim_count) = expect_array_type(&array_type);
    let size_check = gen_array_copy_assignment_size_assertion(assignee, assignee_type, (0..dim_count).map(|d| CudaIdentifier::ArraySizeVar(value.clone(), d)).map(CudaExpression::Identifier), context);
    let copy = gen_array_copy_assignment(assignee, assignee_type, CudaExpression::Identifier(CudaIdentifier::ValueVar(value.clone())), context);
    Ok(Box::new(CudaBlock {
        statements: size_check.chain(std::iter::once(copy)).collect::<Result<Vec<Box<dyn CudaStatement>>, OutputError>>()?
    }))
}

pub fn gen_array_assignment<'stack, 'ast: 'stack>(pos: &TextPosition, assignee: &Name, assignee_type: &Type, value: &Expression, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    let array_type = assignee_type.clone().without_view();
    let (_, dim_count) = expect_array_type(&array_type);
    if assignee_type.is_view() {
        match value {
            Expression::Call(call) => {
                let tmp_var_init = gen_call_array_result_in_tmp_var(pos, &array_type, call, context);
                let size_check = gen_array_copy_assignment_size_assertion(assignee, assignee_type, (0..dim_count).map(|d| CudaIdentifier::TmpSizeVar(d)).map(CudaExpression::Identifier), context);
                let copy = gen_array_copy_assignment(assignee, assignee_type, CudaExpression::Identifier(CudaIdentifier::TmpVar), context);
                Ok(Box::new(CudaBlock {
                    statements: tmp_var_init.chain(size_check).chain(std::iter::once(copy)).collect::<Result<Vec<Box<dyn CudaStatement>>, OutputError>>()?
                }))
            },
            Expression::Variable(var) => match &var.identifier {
                Identifier::BuiltIn(_) => unimplemented!(),
                Identifier::Name(val_name) =>  gen_array_checked_copy_assignment_from_var(pos, assignee, assignee_type, val_name, context)
            },
            Expression::Literal(_) => unimplemented!()
        }
    } else {
        match value {
            Expression::Call(call) => gen_array_move_assignment_from_call(pos, assignee, call, context),
            Expression::Variable(var) => match &var.identifier {
                Identifier::BuiltIn(_) => unimplemented!(),
                Identifier::Name(val_name) => gen_array_checked_copy_assignment_from_var(pos, assignee, assignee_type, val_name, context)
            },
            Expression::Literal(_) => unimplemented!()
        }
    }
}

pub fn gen_assignment<'stack, 'ast: 'stack>(statement: &Assignment, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    match &statement.assignee {
        Expression::Call(call) => {
            if call.function != BuiltInIdentifier::FunctionIndex {
                error_rvalue_not_assignable(statement.pos()).throw()
            }
            Ok(Box::new(gen_scalar_assignment(&statement.assignee, &statement.value, context)?))
        },
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(_) => error_rvalue_not_assignable(statement.pos()).throw(),
            Identifier::Name(name) => {
                let ty = context.calculate_var_type(name, statement.pos());
                if is_mul_var_type(&ty) {
                    gen_array_assignment(statement.pos(), name, &ty, &statement.value, context)
                } else {
                    Ok(Box::new(gen_scalar_assignment(&statement.assignee, &statement.value, context)?))
                }
            }
        }, 
        Expression::Literal(_) => error_rvalue_not_assignable(statement.pos()).throw()
    }
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::Parser;
#[cfg(test)]
use super::context::CudaContextImpl;
#[cfg(test)]
use super::writer::*;

#[test]
fn test_gen_expression() {
    let expr = Expression::parse(&mut fragment_lex("(a + b * c) / d[e,]")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let program = Program { items: vec![] };
    let defs = [
        (Name::l("a"), Type::Primitive(PrimitiveType::Int)),
        (Name::l("b"), Type::Primitive(PrimitiveType::Int,)),
        (Name::l("c"), Type::Primitive(PrimitiveType::Int)),
        (Name::l("d"), Type::Array(PrimitiveType::Int, 1)),
        (Name::l("e"), Type::Primitive(PrimitiveType::Int))
    ];
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::build_for_program(&program).unwrap());
    context.enter_scope(&defs[..]);
    gen_expression(&expr, &mut *context).unwrap().write(&mut writer).unwrap();
    assert_eq!("(a_ + b_ * c_) / d_[e_]", output);
}

#[test]
fn test_gen_expression_function_call() {
    let expr = Expression::parse(&mut fragment_lex("foo[a(b,), c,]")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let program = Program { items: vec![] };
    let defs = [
        (Name::l("a"), Type::Function(vec![Box::new(Type::Array(PrimitiveType::Int, 1))], Some(Box::new(Type::Primitive(PrimitiveType::Int))))),
        (Name::l("foo"), Type::Array(PrimitiveType::Int, 2)),
        (Name::l("b"), Type::Primitive(PrimitiveType::Int)),
        (Name::l("c"), Type::Primitive(PrimitiveType::Int))
    ];
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::build_for_program(&program).unwrap());
    context.enter_scope(&defs[..]);
    gen_expression(&expr, &mut *context).unwrap().write(&mut writer).unwrap();
    assert_eq!("foo_[a_(b_, b_d0) * foo_d1 + c_]", output);
}

#[test]
fn test_gen_expression_pass_index_expression_by_view() {
    let expr = Expression::parse(&mut fragment_lex("foo(a[0,],)")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let program = Program { items: vec![] };
    let defs = [
        (Name::l("foo"), Type::Function(vec![Box::new(Type::View(Box::new(Type::Primitive(PrimitiveType::Int))))], Some(Box::new(Type::Primitive(PrimitiveType::Int))))),
        (Name::l("a"), Type::Array(PrimitiveType::Int, 1))
    ];
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::build_for_program(&program).unwrap());
    context.enter_scope(&defs[..]);
    gen_expression(&expr, &mut *context).unwrap().write(&mut writer).unwrap();
    assert_eq!("foo_(&a_[0])", output);
}

#[test]
fn test_gen_expression_pass_index_expression_by_value() {
    let expr = Expression::parse(&mut fragment_lex("foo(a[0,],)")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let program = Program { items: vec![] };
    let defs = [
        (Name::l("foo"), Type::Function(vec![Box::new(Type::Primitive(PrimitiveType::Int))], Some(Box::new(Type::Primitive(PrimitiveType::Int))))),
        (Name::l("a"), Type::Array(PrimitiveType::Int, 1))
    ];
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::build_for_program(&program).unwrap());
    context.enter_scope(&defs[..]);
    gen_expression(&expr, &mut *context).unwrap().write(&mut writer).unwrap();
    assert_eq!("foo_(a_[0])", output);
}

#[test]
fn test_gen_assignment_array_view_to_array() {
    let assignment = Statement::parse(&mut fragment_lex("a = b;")).unwrap().dynamic_box().downcast::<Assignment>().unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let program = Program { items: vec![] };
    let defs = [
        (Name::l("a"), Type::Array(PrimitiveType::Int, 2)),
        (Name::l("b"), Type::View(Box::new(Type::Array(PrimitiveType::Int, 2))))    
    ];
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::build_for_program(&program).unwrap());
    context.enter_scope(&defs[..]);
    gen_assignment(&*assignment, &mut *context).unwrap().write(&mut writer).unwrap();
    assert_eq!("{
    assert(a_d0 == b_d0);
    assert(a_d1 == b_d1);
    checkCudaStatus(cudaMemcpy(a_, b_, sizeof(int) * a_d0, cudaMemcpyDeviceToDevice));
}", output);
}

#[test]
fn test_gen_assignment_call_result_to_array_view() {
    let assignment = Statement::parse(&mut fragment_lex("b = foo(a, c,);")).unwrap().dynamic_box().downcast::<Assignment>().unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let program = Program { items: vec![] };
    let defs = [
        (Name::l("foo"), Type::Function(vec![
            Box::new(Type::View(Box::new(Type::Array(PrimitiveType::Int, 1)))),
            Box::new(Type::Primitive(PrimitiveType::Int))
        ], Some(Box::new(Type::Array(PrimitiveType::Int, 2))))),
        (Name::l("a"), Type::Array(PrimitiveType::Int, 1)),
        (Name::l("c"), Type::Primitive(PrimitiveType::Int)),
        (Name::l("b"), Type::View(Box::new(Type::Array(PrimitiveType::Int, 2))))
    ];
    let mut context: Box<dyn CudaContext> = Box::new(CudaContextImpl::build_for_program(&program).unwrap());
    context.enter_scope(&defs[..]);
    gen_assignment(&*assignment, &mut *context).unwrap().write(&mut writer).unwrap();
    assert_eq!("{
    int* tmp = nullptr;
    unsigned int tmpd0 = 0;
    unsigned int tmpd1 = 0;
    foo_(a_, a_d0, c_, &tmp, &tmpd0, &tmpd1);
    assert(b_d0 == tmpd0);
    assert(b_d1 == tmpd1);
    checkCudaStatus(cudaMemcpy(b_, tmp, sizeof(int) * b_d0, cudaMemcpyDeviceToDevice));
}", output);
}