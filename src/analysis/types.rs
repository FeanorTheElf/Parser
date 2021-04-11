use super::super::language::prelude::*;
use super::super::language::gwaihir_writer::*;
use super::super::language::concrete_views::*;
use super::topological_sort::call_graph_topological_sort;

fn arithmetic_result_type(pos: &TextPosition, ty1: &Type, ty2: &Type, op: BuiltInIdentifier) -> Result<Type, CompileError> {
    if ty1.is_scalar(PrimitiveType::Int) && ty2.is_scalar(PrimitiveType::Int) {
        Ok(PrimitiveType::Int.scalar(true))
    } else if (ty1.is_scalar(PrimitiveType::Float) || ty1.is_scalar(PrimitiveType::Int)) &&
        (ty2.is_scalar(PrimitiveType::Float) || ty2.is_scalar(PrimitiveType::Int)) 
    {
        Ok(PrimitiveType::Float.scalar(true))
    } else {
        Err(CompileError::bi_operator_not_implemented(pos, &ty1, &ty2, op))
    }
}

impl CompileError {

    fn bi_operator_not_implemented(pos: &TextPosition, ty1: &Type, ty2: &Type, op: BuiltInIdentifier) -> CompileError {
        CompileError::new(
            pos,
            format!("Operator {} not implemented for {} and {}", op.get_symbol(), DisplayWrapper::from(ty1), DisplayWrapper::from(ty2)),
            ErrorType::TypeError
        )
    }

    fn un_operator_not_implemented(pos: &TextPosition, ty: &Type, op: BuiltInIdentifier) -> CompileError {
        CompileError::new(
            pos,
            format!("Operator {} not implemented for {}", op.get_symbol(), DisplayWrapper::from(ty)),
            ErrorType::TypeError
        )
    }

    fn type_error(pos: &TextPosition, expected: &Type, actual: &Type) -> CompileError {
        unimplemented!()
    }

    fn wrong_param_count(pos: &TextPosition, expected: usize, actual: usize) -> CompileError {
        unimplemented!()
    }

    fn not_callable(pos: &TextPosition, ty: &Type) -> CompileError {
        CompileError::new(
            pos, 
            format!("Type {} not callable", DisplayWrapper::from(ty)),
            ErrorType::TypeError
        )
    }

    fn expected_nonvoid(pos: &TextPosition) -> CompileError {
        CompileError::new(
            pos, 
            format!("Value expected, but got void"),
            ErrorType::TypeError
        )
    }
}

///
/// Calculates the result type of the builtin operator applied to parameters of the given types.
/// This also performs type checking, i.e. will return a compile error if the types do not match within the expression.
/// Created views have concrete view information set.
/// This function completely consumes the passed param_types iterator (useful if the iterator has side effects).
/// 
fn calculate_builtin_call_result_type<'a, I>(op: BuiltInIdentifier, pos: &TextPosition, mut param_types: I) -> Result<Type, CompileError>
    where I: Iterator<Item = (Result<&'a Type, CompileError>, &'a TextPosition)>
{
    match op {
        BuiltInIdentifier::ViewZeros => {
            let mut dimension_count = 0;
            for param in param_types {
                let param_type = param.0?;
                if !param_type.is_scalar(PrimitiveType::Int) {
                    return Err(CompileError::type_error(param.1, &PrimitiveType::Int.scalar(true), param_type));
                }
                dimension_count += 1;
            }
            return Ok(PrimitiveType::Int.array(dimension_count, false).with_concrete_view(VIEW_ZEROS));
        },
        BuiltInIdentifier::FunctionAdd |
            BuiltInIdentifier::FunctionMul => 
        {
            return param_types.try_fold(PrimitiveType::Int.scalar(true), |current, (try_type, pos)| {
                let param_type = try_type?;
                arithmetic_result_type(pos, &current, &param_type, op)
            });
        },
        BuiltInIdentifier::FunctionAnd |
            BuiltInIdentifier::FunctionOr => 
        {
            for param in param_types {
                let param_type = param.0?;
                if !param_type.is_scalar(PrimitiveType::Bool) {
                    return Err(CompileError::bi_operator_not_implemented(param.1, &PrimitiveType::Bool.scalar(true), &param_type, op));
                }
            }
            return Ok(PrimitiveType::Bool.scalar(true));
        },
        BuiltInIdentifier::FunctionEq |
            BuiltInIdentifier::FunctionGeq |
            BuiltInIdentifier::FunctionLeq |
            BuiltInIdentifier::FunctionNeq |
            BuiltInIdentifier::FunctionLs |
            BuiltInIdentifier::FunctionGt => 
        {
            param_types.try_fold(PrimitiveType::Int.scalar(true), |current, (try_type, pos)| {
                let param_type = try_type?;
                arithmetic_result_type(pos, &current, param_type, op)
            })?;
            return Ok(PrimitiveType::Bool.scalar(true));
        },
        BuiltInIdentifier::FunctionIndex => {
            let (array_type, index_pos) = param_types.next().expect("index function call has no parameters, but the indexed array should be the first parameter");
            let mut count = 0;
            for param in param_types {
                let param_type = param.0?;
                if !param_type.is_scalar(PrimitiveType::Int) {
                    return Err(CompileError::type_error(param.1, &PrimitiveType::Int.scalar(true), param_type));
                }
                count += 1;
            }
            match array_type? {
                Type::Static(static_type) => {
                    if static_type.dims != count {
                        return Err(CompileError::wrong_param_count(index_pos, static_type.dims, count));
                    } else {
                        return Ok(static_type.base.scalar(static_type.is_mutable()).with_concrete_view(VIEW_INDEX));
                    }
                },
                Type::View(view_type) => {
                    if view_type.view_onto.dims != count {
                        return Err(CompileError::wrong_param_count(index_pos, view_type.view_onto.dims, count));
                    } else {
                        return Ok(
                            view_type.view_onto.base.scalar(view_type.view_onto.is_mutable())
                                .with_concrete_view_dyn(ViewComposed::compose(
                                    VIEW_INDEX, 
                                    view_type.concrete_view.as_ref().expect("Concrete view of dependency not available, current calculation only works with linear dependencies").clone()
                                ))
                        );
                    }
                }
            };
        },
        BuiltInIdentifier::FunctionUnaryDiv | BuiltInIdentifier::FunctionUnaryNeg => {
            let (base_ty, pos) = param_types.next().expect("unary division and negation has no parameters, but it must be applied to a value");
            if param_types.next().is_some() {
                panic!("unary operator has more than one argument");
            }
            if !base_ty?.is_scalar(PrimitiveType::Int) && !base_ty?.is_scalar(PrimitiveType::Float) {
                return Err(CompileError::un_operator_not_implemented(pos, base_ty?, op));
            } else {
                return base_ty.map(Type::clone);
            }
        }
    }
}

///
/// This also performs type checking, i.e. will return a compile error if the types do not match within the expression.
/// 
/// This also consumes the param_types iterator.
///
fn calculate_defined_function_call_result_type<'a, I>(
    call: &FunctionCall, 
    function_type: &Type, 
    mut param_types: I
) -> Result<Option<Type>, CompileError> 
where I: Iterator<Item = (Result<&'a Type, CompileError>, &'a TextPosition)>
{
    let function_type = if let Some(ty) = function_type.as_function() {
        ty
    } else {
        return Err(CompileError::not_callable(call.pos(), function_type))
    };
    for (given_param, expected_param) in param_types.zip(function_type.parameter_types()) {
        let param_type = given_param.0?;
        if !param_type.is_implicitly_convertable(expected_param) && !param_type.is_viewable_as(expected_param) {
            return Err(CompileError::type_error(given_param.1, expected_param, param_type));
        }
    }
    assert!(function_type.is_void() || !function_type.return_type().unwrap().is_view());

    return Ok(function_type.return_type().cloned());
}

///
/// Calculates the result type of the given expression, including correct concrete view types. If the
/// expression is a function call expression, the calculated type pointer will also be written into 
/// the type cache of that node. The parameter types of a function call will always be computed and
/// stored in the corresponding cache.
/// 
/// This will set concrete view types.
/// 
/// This also performs type checking, i.e. will return a compile error if the types do not match within the expression.
/// 
fn calculate_and_store_type<'a>(expr: &'a mut Expression, scopes: &DefinitionScopeStackMut) -> Result<&'a Option<Type>, CompileError> {
    let result = match expr {
        Expression::Call(call) => {
            if call.result_type_cache.is_some() {
                return Ok(call.result_type_cache.as_ref().unwrap());
            }
            let mut parameter_types = call.parameters.iter_mut()
                    .map(|p| (calculate_and_store_type_nonvoid(p, scopes), p.pos())).fuse();

            if let Expression::Variable(var) = &call.function {
                if let Identifier::BuiltIn(op) = &var.identifier {
                    call.result_type_cache = Some(
                        Some(calculate_builtin_call_result_type(*op, call.pos(), parameter_types)?)
                    );
                } else {
                    let called_function_type = calculate_and_store_type(&mut call.function, scopes)?.unwrap();
                    call.result_type_cache = Some(
                        calculate_defined_function_call_result_type(call, &called_function_type, parameter_types)?
                    )
                }
            } else {
                let called_function_type = calculate_and_store_type(&mut call.function, scopes)?.unwrap();
                call.result_type_cache = Some(
                    calculate_defined_function_call_result_type(call, &called_function_type, parameter_types)?
                )
            };
            assert!(parameter_types.next().is_none());
            return Ok(call.result_type_cache.as_ref().unwrap());
        },
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => unimplemented!(),
            Identifier::BuiltIn(op) => panic!("Called calculate_type() on builtin identifier {:?}, but builtin identifiers can have different types depending on context", op)
        },
        Expression::Literal(lit) => unimplemented!()
    };
}

fn calculate_and_store_type_nonvoid<'a>(
    expr: &'a mut Expression, 
    scopes: &DefinitionScopeStackMut
) -> Result<&'a Type, CompileError> {
    if let Some(ty) = calculate_and_store_type(expr, scopes)? {
        Ok(ty)
    } else {
        Err(CompileError::expected_nonvoid(expr.pos()))
    }
}

pub fn determine_types_in_function(
    function: &mut Function,
    global_scope: &DefinitionScopeStackMut
) -> Result<(), CompileError> {
    function.traverse_preorder_mut(global_scope, &mut |block, parent_scopes| {
        let mut scopes = parent_scopes.child_stack();
        for statement in block.statements_mut() {
            for expression in statement.expressions_mut() {
                calculate_and_store_type(expression, &scopes);
            }
            if let Some(def) = statement.as_sibling_symbol_definition_mut() {
                // if we have backward visible declarations here, we need to watch out for dependency order,
                // so everything will get a good deal more complicated
                assert!(!def.is_backward_visible());
                scopes.register(def.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic_mut(def));
            }
        }
        return RECURSE;
    })
}

pub fn determine_types_in_program(program: &mut Program) -> Result<(), CompileError> {
    program.for_functions_mut(&mut |func, _scopes| {
        let mut counter = 0;
        for p in &mut func.parameters {
            if p.get_type().is_view() {
                p.var_type.as_view_mut().unwrap().concrete_view = Some(Box::new(ViewTemplate::new(counter)));
                counter += 1;
            }
        }
        return Ok(());
    })?;
    let mut global_scope = DefinitionScopeStackMut::new();
    for function in call_graph_topological_sort(program)? {

        global_scope.register(function.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic_mut(function));
    }
    return Ok(());
}

pub trait TypeStored {
    fn get_stored_voidable_type(&self) -> Option<&Type>;

    fn get_stored_type(&self) -> &Type {
        self.get_stored_voidable_type().unwrap()
    }
}

impl TypeStored for FunctionCall {
    fn get_stored_voidable_type(&self) -> Option<&Type> {
        self.result_type_cache.as_ref().expect("Called get_type() on a function call expression whose type has not yet been calculated").as_ref()
    }
}

impl TypeStored for Literal {
    fn get_stored_voidable_type(&self) -> Option<&Type> {
        Some(&self.literal_type)
    }
}

pub fn get_expression_type(expr: &Expression, scopes: &DefinitionScopeStack) -> Option<Type> {
    match expr {
        Expression::Call(call) => {
            call.get_stored_voidable_type().map(Type::clone)
        },
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => Some(scopes.get_defined(name, var.pos()).internal_error().get_type()),
            Identifier::BuiltIn(op) => panic!("Called calculate_type() on builtin identifier {:?}, but builtin identifiers can have different types depending on context", op)
        },
        Expression::Literal(lit) => lit.get_stored_voidable_type().map(Type::clone)
    }
}

#[test]
fn test_determine_types_in_program() {
    let mut program = Program::parse(&mut lex_str("
    
    fn foo(x: &int[,],) {
        let a: &int[,] = x;
        let b: &int[,] = zeros(5, 8,);
    }
    
    ")).unwrap();

    determine_types_in_program(&mut program).internal_error();

    assert_eq!(
        &Template::new(0) as &dyn ConcreteView,
        program.items[0].body.as_ref().unwrap().statements[0]
            .downcast::<LocalVariableDeclaration>().unwrap()
            .declaration.variable_type.deref(program.lifetime())
            .unwrap_view().get_concrete().unwrap()
    );

    assert_eq!(
        &ZeroView::new() as &dyn ConcreteView,
        program.items[0].body.as_ref().unwrap().statements[1]
            .downcast::<LocalVariableDeclaration>().unwrap()
            .declaration.variable_type.deref(program.lifetime())
            .unwrap_view().get_concrete().unwrap()
    );
}