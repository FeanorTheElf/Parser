use super::super::language::prelude::*;
use super::super::language::gwaihir_writer::*;
use super::super::language::concrete_views::*;
use super::statement_types::typecheck_statement;
use super::topological_sort;

fn arithmetic_result_type(pos: &TextPosition, ty1: &Type, ty2: &Type, op: BuiltInIdentifier) -> Result<Type, CompileError> {
    if ty1.is_scalar_of(PrimitiveType::Int) && ty2.is_scalar_of(PrimitiveType::Int) {
        Ok(SCALAR_INT)
    } else if (ty1.is_scalar_of(PrimitiveType::Float) || ty1.is_scalar_of(PrimitiveType::Int)) &&
        (ty2.is_scalar_of(PrimitiveType::Float) || ty2.is_scalar_of(PrimitiveType::Int)) 
    {
        Ok(SCALAR_FLOAT)
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
        CompileError::new(
            pos,
            format!("Expected type {} but got type {}", DisplayWrapper::from(expected), DisplayWrapper::from(actual)),
            ErrorType::TypeError
        )
    }

    fn wrong_param_count(pos: &TextPosition, expected: usize, actual: usize) -> CompileError {
        CompileError::new(
            pos,
            format!("Wrong parameter count, expected {} parameters but got {}", expected, actual),
            ErrorType::TypeError
        )
    }

    fn not_callable(pos: &TextPosition, ty: &Type) -> CompileError {
        CompileError::new(
            pos, 
            format!("Type {} not callable", DisplayWrapper::from(ty)),
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
fn calculate_builtin_call_result_type<'a, I>(op: BuiltInIdentifier, mut param_types: I) -> Result<Type, CompileError>
    where I: Iterator<Item = (Result<&'a Type, CompileError>, TextPosition)>
{
    match op {
        BuiltInIdentifier::ViewZeros => {
            let mut dimension_count = 0;
            for param in param_types {
                let param_type = param.0?;
                if !param_type.is_scalar_of(PrimitiveType::Int) {
                    return Err(CompileError::type_error(&param.1, &SCALAR_INT, param_type));
                }
                dimension_count += 1;
            }
            return Ok(PrimitiveType::Int.array(dimension_count, false).with_concrete_view(VIEW_ZEROS));
        },
        BuiltInIdentifier::FunctionAdd |
            BuiltInIdentifier::FunctionMul => 
        {
            return param_types.try_fold(SCALAR_INT, |current, (try_type, pos)| {
                let param_type = try_type?;
                arithmetic_result_type(&pos, &current, &param_type, op)
            });
        },
        BuiltInIdentifier::FunctionAnd |
            BuiltInIdentifier::FunctionOr => 
        {
            for param in param_types {
                let param_type = param.0?;
                if !param_type.is_scalar_of(PrimitiveType::Bool) {
                    return Err(CompileError::bi_operator_not_implemented(&param.1, &SCALAR_BOOL, &param_type, op));
                }
            }
            return Ok(SCALAR_BOOL);
        },
        BuiltInIdentifier::FunctionEq |
            BuiltInIdentifier::FunctionGeq |
            BuiltInIdentifier::FunctionLeq |
            BuiltInIdentifier::FunctionNeq |
            BuiltInIdentifier::FunctionLs |
            BuiltInIdentifier::FunctionGt => 
        {
            param_types.try_fold(SCALAR_INT, |current, (try_type, pos)| {
                let param_type = try_type?;
                arithmetic_result_type(&pos, &current, param_type, op)
            })?;
            return Ok(SCALAR_BOOL);
        },
        BuiltInIdentifier::FunctionIndex => {
            let (try_array_type, index_pos) = param_types.next().expect("index function call has no parameters, but the indexed array should be the first parameter");
            let mut count = 0;
            for param in param_types {
                let param_type = param.0?;
                if !param_type.is_scalar_of(PrimitiveType::Int) {
                    return Err(CompileError::type_error(&param.1, &SCALAR_INT, param_type));
                }
                count += 1;
            }
            let array_type = try_array_type?;
            match array_type {
                Type::Static(static_type) => {
                    if static_type.dims != count {
                        return Err(CompileError::wrong_param_count(&index_pos, static_type.dims, count));
                    } else {
                        return Ok(static_type.base.as_scalar_type(static_type.is_mutable()).with_concrete_view(VIEW_INDEX));
                    }
                },
                Type::View(view_type) => {
                    if view_type.view_onto.dims != count {
                        return Err(CompileError::wrong_param_count(&index_pos, view_type.view_onto.dims, count));
                    } else {
                        return Ok(
                            view_type.view_onto.base.as_scalar_type(view_type.view_onto.is_mutable())
                                .with_concrete_view_dyn(ViewComposed::compose(
                                    VIEW_INDEX, 
                                    view_type.concrete_view.as_ref().expect("Concrete view of dependency not available, current calculation only works with linear dependencies").clone()
                                ))
                        );
                    }
                },
                Type::Function(_) => {
                    return Err(CompileError::un_operator_not_implemented(&index_pos, array_type, op));
                }
            };
        },
        BuiltInIdentifier::FunctionUnaryDiv | BuiltInIdentifier::FunctionUnaryNeg => {
            let (try_base_type, pos) = param_types.next().expect("unary division and negation has no parameters, but it must be applied to a value");
            let base_type = try_base_type?;
            if param_types.next().is_some() {
                panic!("unary operator has more than one argument");
            }
            if !base_type.is_scalar_of(PrimitiveType::Int) && !base_type.is_scalar_of(PrimitiveType::Float) {
                return Err(CompileError::un_operator_not_implemented(&pos, base_type, op));
            } else {
                return Ok(base_type.clone());
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
    call_pos: &TextPosition, 
    function_type: &Type, 
    param_types: I
) -> Result<Option<Type>, CompileError> 
where I: Iterator<Item = (Result<&'a Type, CompileError>, TextPosition)>
{
    let function_type = if let Some(ty) = function_type.as_function() {
        ty
    } else {
        return Err(CompileError::not_callable(call_pos, function_type))
    };
    for (given_param, expected_param) in param_types.zip(function_type.parameter_types()) {
        let param_type = given_param.0?;
        if !param_type.is_implicitly_convertable(expected_param) && !param_type.is_viewable_as(expected_param) {
            return Err(CompileError::type_error(&given_param.1, expected_param, param_type));
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
fn calculate_and_store_type<'a>(expr: &'a mut Expression, scopes: &'a DefinitionScopeStackMut<'_, 'a>) -> Result<Option<&'a Type>, CompileError> {
    match expr {
        Expression::Call(call) => {
            if call.result_type_cache.is_some() {
                return Ok(call.result_type_cache.as_ref().unwrap().as_ref());
            }
            let mut parameter_types = call.parameters.iter_mut()
                    .map(|p| {
                        let pos = p.pos().clone();
                        (calculate_and_store_type_nonvoid(p, scopes), pos)
                    }).fuse();

            if let Expression::Variable(var) = &call.function {
                if let Identifier::BuiltIn(op) = &var.identifier {
                    call.result_type_cache = Some(
                        Some(calculate_builtin_call_result_type(*op, parameter_types.by_ref())?)
                    );
                } else {
                    let called_function_type = calculate_and_store_type(&mut call.function, scopes)?.unwrap();
                    call.result_type_cache = Some(
                        calculate_defined_function_call_result_type(&call.pos, &called_function_type, parameter_types.by_ref())?
                    )
                }
            } else {
                let called_function_type = calculate_and_store_type(&mut call.function, scopes)?.unwrap();
                call.result_type_cache = Some(
                    calculate_defined_function_call_result_type(&call.pos, &called_function_type, parameter_types.by_ref())?
                )
            };
            assert!(parameter_types.next().is_none());
            return Ok(call.result_type_cache.as_ref().unwrap().as_ref());
        },
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => return Ok(Some(&scopes.get_defined(name, var.pos())?.get_type())),
            Identifier::BuiltIn(op) => panic!("Called calculate_type() on builtin identifier {:?}, but builtin identifiers can have different types depending on context", op)
        },
        Expression::Literal(lit) => return Ok(Some(&lit.literal_type))
    };
}

fn calculate_and_store_type_nonvoid<'a>(
    expr: &'a mut Expression, 
    scopes: &'a DefinitionScopeStackMut<'_, 'a>
) -> Result<&'a Type, CompileError> {
    let position = expr.pos().clone();
    if let Some(ty) = calculate_and_store_type(expr, scopes)? {
        return Ok(ty);
    }
    return Err(CompileError::expected_nonvoid(&position));
}

fn determine_types_in_statement(
    statement: &mut dyn Statement,
    scopes: &DefinitionScopeStackMut,
    function_type: &FunctionType
) -> Result<(), CompileError> {
    for expression in statement.expressions_mut() {
        calculate_and_store_type(expression, &scopes)?;
    }
    typecheck_statement(statement, scopes, function_type)?;
    return Ok(());
}

pub fn determine_types_in_function(
    function: &mut Function,
    global_scope: &DefinitionScopeStackMut
) -> Result<(), CompileError> {
    function.traverse_preorder_block_mut(global_scope, &mut |block, parent_scopes, function_type| {
        let mut scopes = parent_scopes.child_stack();
        for statement in block.statements_mut() {
            determine_types_in_statement(statement, &scopes, function_type)?;
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
                p.get_type_mut().as_view_mut().unwrap().concrete_view = Some(Box::new(ViewTemplate::new(counter)));
                counter += 1;
            }
        }
        return Ok(()) as Result<(), !>;
    }).unwrap();
    program.for_functions_ordered_mut(
        |prog| topological_sort::call_graph_topological_sort(prog),
        &mut |function, global_scope| determine_types_in_function(function, global_scope)
    )?;
    return Ok(());
}


#[test]
fn test_determine_types_in_program() {
    let mut program = Program::parse(&mut lex_str("
    
    fn foo(x: &int[,],) {
        let a: &int[,] init x;
        let b: &int[,,] init zeros(5, 8,);
    }
    
    ")).unwrap();

    determine_types_in_program(&mut program).internal_error();

    assert_eq!(
        &ViewTemplate::new(0) as &dyn View,
        program.items[0].body.as_ref().unwrap()
            .statements().next().unwrap()
            .downcast::<LocalVariableDeclaration>().unwrap()
            .declaration.get_type()
            .as_view().unwrap()
            .get_concrete()
    );

    assert_eq!(
        &VIEW_ZEROS as &dyn View,
        program.items[0].body.as_ref().unwrap()
            .statements().skip(1).next().unwrap()
            .downcast::<LocalVariableDeclaration>().unwrap()
            .declaration.get_type()
            .as_view().unwrap()
            .get_concrete()
    );
}

#[test]
fn test_typecheck_assignment() {
    let mut program = Program::parse(&mut lex_str("
    
    fn test(x: &int[,],) {
        let a: int[,] init x;
        a[1,] = x[1,];
    }
    
    ")).unwrap();

    assert!(*determine_types_in_program(&mut program).unwrap_err().error_type() == ErrorType::TypeError);
    
    let mut program = Program::parse(&mut lex_str("
    
    fn test(x: &int[,],) {
        let a: write int[,] init x;
        a[1,] = x[1,];
    }
    
    ")).unwrap();

    determine_types_in_program(&mut program).unwrap();
}

#[test]
fn test_typecheck_return() {
    let mut program = Program::parse(&mut lex_str("
    
    fn test(x: &int[,],): int[,] {
        let a: write int[,] init x;
        a[1,] = 1;
        return 1;
    }
    
    ")).unwrap();

    assert!(*determine_types_in_program(&mut program).unwrap_err().error_type() == ErrorType::TypeError);
    
    let mut program = Program::parse(&mut lex_str("
    
    fn test(x: &int[,],): int[,] {
        let a: write int[,] init x;
        a[1,] = 1;
        return a;
    }
    
    ")).unwrap();

    determine_types_in_program(&mut program).unwrap();
}