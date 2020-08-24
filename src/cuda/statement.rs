use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::super::analysis::type_error::*;
use super::CudaContext;
use super::ast::*;

fn error_call_dynamic_expression(pos: &TextPosition) -> OutputError {
    OutputError::UnsupportedCode(pos.clone(), format!("Calling dynamic expressions is not supported by the Cuda Backend"))
}

fn is_mul_var_type(ty: &Type) -> bool {
    match ty {
        Type::Array(_, dim) => *dim > 0,
        Type::View(viewn) => is_mul_var_type(viewn),
        _ => false
    }
}

fn is_generated_with_output_parameter(func: &Function) -> bool {
    func.return_type.as_ref().map(is_mul_var_type).unwrap_or(false)
}

pub fn gen_primitive_type(value: &PrimitiveType, ptr_count: u32) -> CudaType {
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
        _ => panic!("expect_array_type() called for non-array type")
    }
}

pub fn gen_variables(pos: &TextPosition, var: &Name, ty: &Type) -> impl Iterator<Item = (CudaType, CudaIdentifier)> {
    let (base, dim_count) = match ty {
        Type::Array(base, dim) => (gen_primitive_type(base, 1), *dim),
        Type::Function(_, _) => panic!("Cannot represent a function in cuda variables"),
        Type::JumpLabel => panic!("Cannot represent a jump label in cuda variables"),
        Type::Primitive(base) => (gen_primitive_type(base, 0), 0),
        Type::TestType => error_test_type(pos),
        Type::View(viewn) => match &**viewn {
            Type::Array(base, dim) => (gen_primitive_type(base, 1), *dim),
            Type::Function(_, _) => panic!("Cannot represent a function in cuda variables"),
            Type::JumpLabel => panic!("Cannot represent a jump label in cuda variables"),
            Type::Primitive(base) => (gen_primitive_type(base, 0), 1),
            Type::TestType => error_test_type(pos),
            Type::View(_) => error_nested_view(pos).throw()
        }
    };
    let var_copy = var.clone();
    std::iter::once((base, CudaIdentifier::ValueVar(var.clone()))).chain(
        (0..dim_count).map(move |d| (CudaType {
            ptr_count: 0,
            constant: true,
            base: CudaPrimitiveType::Index
        }, CudaIdentifier::ArraySizeVar(var_copy.clone(), d))))
}

fn gen_variables_as_output_params(pos: &TextPosition, var: &Name, ty: &Type) -> impl Iterator<Item = (CudaType, CudaExpression)> {
    gen_variables(pos, var, ty).map(|(ty, var)| (CudaType {
        base: ty.base,
        constant: ty.constant,
        ptr_count: ty.ptr_count + 1
    }, CudaExpression::AddressOf(Box::new(CudaExpression::Identifier(var)))))
}

fn gen_defined_function_call<'stack, 'ast: 'stack, I>(call: &FunctionCall, func: &Function, mut output_params: I, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<CudaExpression, OutputError> 
    where I: Iterator<Item = CudaExpression>
{
    if !is_generated_with_output_parameter(func) {
        assert!(output_params.next().is_none());
    }
    let params = call.parameters.iter().zip(func.params.iter()).flat_map(|(param, formal_param)| match param {
        Expression::Call(subcall) => Box::new(std::iter::once(gen_function_call(subcall, std::iter::empty(), context))) as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>,
        Expression::Literal(_lit) => unimplemented!(),
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => Box::new(gen_variables(var.pos(), name, &formal_param.variable_type).map(|(_ty, p)| Ok(CudaExpression::Identifier(p)))) as Box<dyn Iterator<Item = Result<CudaExpression, OutputError>>>,
            Identifier::BuiltIn(_) => unimplemented!()
        }
    }).chain(output_params.map(|o| Ok(o)));
    Ok(CudaExpression::Call(CudaIdentifier::ValueVar(func.identifier.clone()), params.collect::<Result<Vec<CudaExpression>, OutputError>>()?))
}

fn gen_index_expression<'a, 'stack, 'ast: 'stack, I>(array: &Name, indices: I, context: &mut dyn CudaContext<'stack, 'ast>) -> impl Iterator<Item = Result<CudaExpression, OutputError>> 
    where I: Iterator<Item = &'a Expression>
{
    let index = indices.enumerate().map(|(dim, index)| Ok((AddSub::Plus, CudaExpression::Product(vec![
        (MulDiv::Multiply, gen_expression(index, context)?),
        (MulDiv::Multiply, CudaExpression::Identifier(CudaIdentifier::ArraySizeVar(array.clone(), dim as u32)))
    ])))).collect::<Result<Vec<(AddSub, CudaExpression)>, OutputError>>();

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
                Identifier::Name(name) => gen_index_expression(name, call.parameters.iter().skip(1), context).next().unwrap()?
            }
        }
    })
}

fn gen_function_call<'stack, 'ast: 'stack, I>(call: &FunctionCall, output_params: I, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<CudaExpression, OutputError> 
    where I: Iterator<Item = CudaExpression>
{
    match &call.function {
        Expression::Call(_) => Err(error_call_dynamic_expression(call.pos())),
        Expression::Literal(_) => error_not_callable(call.pos(), &Type::Primitive(PrimitiveType::Int)).throw(),
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(_) => unimplemented!(),
            Identifier::Name(name) => {
                let function = context.get_scopes().get_defined(name, call.pos()).unwrap().dynamic().downcast_ref::<Function>().unwrap();
                gen_defined_function_call(call, function, output_params, context)
            }
        }
    }
}

fn gen_expression<'stack, 'ast: 'stack>(expr: &Expression, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<CudaExpression, OutputError>  {
    match expr {
        Expression::Call(call) => gen_function_call(call, std::iter::empty(), context),
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(_) => unimplemented!(),
            Identifier::Name(name) => {
                let mut result = gen_variables(expr.pos(), name, &context.calculate_var_type(name));
                let r = result.next().unwrap();
                assert!(result.next().is_none());
                Ok(CudaExpression::Identifier(r.1))
            }
        },
        Expression::Literal(_) => unimplemented!()
    }
}

fn gen_array_copy_assignment<'stack, 'ast: 'stack>(assignee: &Name, assignee_type: &Type, value: CudaExpression, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    let (base_type, _) = expect_array_type(assignee_type);
    let destination = if assignee_type.is_view() {
        CudaExpression::Identifier(CudaIdentifier::ValueVar(assignee.clone()))
    } else {
        CudaExpression::AddressOf(Box::new(CudaExpression::Identifier(CudaIdentifier::ValueVar(assignee.clone()))))
    };
    Ok(Box::new(CudaMemcpy {
        destination: destination,
        source: value,
        device: context.is_device_context(),
        length: CudaExpression::Identifier(CudaIdentifier::ArraySizeVar(assignee.clone(), 0)),
        base_type: gen_primitive_type(&base_type, 0)
    }))
}

fn gen_array_move_assignment_from_call<'stack, 'ast: 'stack>(pos: &TextPosition, assignee: &Name, value: &FunctionCall, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    let ty = context.calculate_var_type(assignee);
    Ok(Box::new(gen_function_call(value, gen_variables_as_output_params(pos, assignee, &ty).map(|(_, v)| v), context)?))
}

fn gen_array_move_assignment_from_var<'stack, 'ast: 'stack>(_pos: &TextPosition, assignee: &Name, value: &Name, context: &mut dyn CudaContext<'stack, 'ast>) -> impl Iterator<Item = Result<Box<dyn CudaStatement>, OutputError>> {
    let ty = context.calculate_var_type(assignee);
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
        var_type: gen_primitive_type(&base_type, 1)
    }).chain((0..dim).map(|d| CudaVarDeclaration {
        var: CudaIdentifier::TmpSizeVar(d),
        value: Some(CudaExpression::Literal(0)),
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

pub fn gen_array_assignment<'stack, 'ast: 'stack>(pos: &TextPosition, assignee: &Name, assignee_type: &Type, value: &Expression, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    if assignee_type.is_view() {
        match value {
            Expression::Call(call) => {
                let tmp_var_init = gen_call_array_result_in_tmp_var(pos, &assignee_type.clone().without_view(), call, context);
                let copy = gen_array_copy_assignment(assignee, assignee_type, CudaExpression::Identifier(CudaIdentifier::TmpVar), context);
                Ok(Box::new(CudaBlock {
                    statements: tmp_var_init.chain(std::iter::once(copy)).collect::<Result<Vec<Box<dyn CudaStatement>>, OutputError>>()?
                }))
            },
            Expression::Variable(var) => match &var.identifier {
                Identifier::BuiltIn(_) => unimplemented!(),
                Identifier::Name(val_name) =>  gen_array_copy_assignment(assignee, assignee_type, CudaExpression::Identifier(CudaIdentifier::ValueVar(val_name.clone())), context)
            },
            Expression::Literal(_) => unimplemented!()
        }
    } else {
        match value {
            Expression::Call(call) => gen_array_move_assignment_from_call(pos, assignee, call, context),
            Expression::Variable(var) => match &var.identifier {
                Identifier::BuiltIn(_) => unimplemented!(),
                Identifier::Name(value_name) => Ok(Box::new(CudaBlock {
                    statements: gen_array_move_assignment_from_var(pos, assignee, value_name, context).collect::<Result<Vec<Box<dyn CudaStatement>>, OutputError>>()?
                }))
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
                let ty = context.calculate_var_type(name);
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