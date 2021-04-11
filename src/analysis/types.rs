use super::super::language::prelude::*;
use super::super::language::gwaihir_writer::*;
use super::super::language::concrete_views::*;
use super::topological_sort::call_graph_topological_sort;

fn is_arithmetic(ty: PrimitiveType) -> bool {
    ty == PrimitiveType::Int || ty == PrimitiveType::Float
}

fn arithmetic_result_type(pos: &TextPosition, ty1: Type, ty2: Type, op: BuiltInIdentifier) -> Result<Type, CompileError> {
    match (ty1.as_scalar(), ty2.as_scalar()) {
        (Some(prim1), Some(prim2)) if is_arithmetic(*prim1) && is_arithmetic(*prim2) => {
            if *prim1 == PrimitiveType::Int && *prim2 == PrimitiveType::Int {
                Ok(SCALAR_INT)
            } else {
                Ok(SCALAR_FLOAT)
            }
        },
        (_, _) => {
            Err(CompileError::bi_operator_not_implemented(pos, &ty1, &ty2, op))
        }
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

    fn type_error(pos: &TextPosition, expected: Type, actual: Type) -> CompileError {
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
}

///
/// Calculates the result type of the builtin operator applied to parameters of the given types.
/// This also performs type checking, i.e. will return a compile error if the types do not match within the expression.
/// Created views have concrete view information set.
/// This function completely consumes the passed param_types iterator (useful if the iterator has side effects).
/// 
fn calculate_builtin_call_result_type<'a, I>(op: BuiltInIdentifier, pos: &TextPosition, mut param_types: I) -> Result<Type, CompileError>
    where I: Iterator<Item = (Result<Type, CompileError>, &'a TextPosition)>
{
    match op {
        BuiltInIdentifier::ViewZeros => {
            let mut dimension_count = 0;
            for param in param_types {
                let param_type = param.0?;
                if param_type != PrimitiveType::Int {
                    return Err(CompileError::type_error(param.1, SCALAR_INT, param_type));
                }
                dimension_count += 1;
            }
            return Ok(PrimitiveType::Int.array(dimension_count).with_concrete_view(VIEW_ZEROS));
        },
        BuiltInIdentifier::FunctionAdd |
            BuiltInIdentifier::FunctionMul => 
        {
            return param_types.try_fold(SCALAR_INT, |current, (try_type, pos)| {
                let param_type = try_type?;
                arithmetic_result_type(pos, current, param_type, op)
            });
        },
        BuiltInIdentifier::FunctionAnd |
            BuiltInIdentifier::FunctionOr => 
        {
            for param in param_types {
                let param_type = param.0?;
                if param_type != PrimitiveType::Bool {
                    return Err(CompileError::bi_operator_not_implemented(param.1, &SCALAR_BOOL, &param_type, op));
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
                arithmetic_result_type(pos, current, param_type, op)
            })?;
            return Ok(SCALAR_BOOL);
        },
        BuiltInIdentifier::FunctionIndex => {
            let (array_type, index_pos) = param_types.next().expect("index function call has no parameters, but the indexed array should be the first parameter");
            let mut count = 0;
            for param in param_types {
                let param_type = param.0?;
                if param_type != PrimitiveType::Int {
                    return Err(CompileError::type_error(param.1, SCALAR_INT, param_type));
                }
                count += 1;
            }
            match array_type? {
                Type::Static(static_type) => {
                    if static_type.dims != count {
                        return Err(CompileError::wrong_param_count(index_pos, static_type.dims, count));
                    } else {
                        return Ok(static_type.base.scalar().with_view());
                    }
                },
                Type::View(view_type) => {
                    if view_type.view_onto.dims != count {
                        return Err(CompileError::wrong_param_count(index_pos, view_type.view_onto.dims, count));
                    } else {
                        return Ok(view_type.view_onto.base.scalar().with_view());
                    }
                }
            };
        },
        BuiltInIdentifier::FunctionUnaryDiv | BuiltInIdentifier::FunctionUnaryNeg => {
            let (base_ty, pos) = param_types.next().expect("unary division and negation has no parameters, but it must be applied to a value");
            if param_types.next().is_some() {
                panic!("unary operator has more than one argument");
            }
            if base_ty?.as_scalar().map(|ty| !is_arithmetic(*ty)).unwrap_or(true) {
                return Err(CompileError::un_operator_not_implemented(pos, &base_ty?, op));
            } else {
                return base_ty;
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
where I: Iterator<Item = (Result<Type, CompileError>, &'a TextPosition)>
{
    let function_type = if let Some(ty) = function_type.as_function() {
        ty
    } else {
        return Err(CompileError::not_callable(call.pos(), function_type))
    };
    for (given_param, expected_param) in param_types.zip(function_type.parameter_types()) {
        let param_type = given_param.0?;
        if !param_type.is_implicitly_convertable(expected_param) {
            return Err(error_type_not_convertable(given_param.1, param_type, *expected_param, types.get_lifetime()));
        }
    }
    debug_assert!(function_type.return_type.is_void() || !function_type.return_type.unwrap().deref(types.get_lifetime()).is_view());

    // clone the type here, as it is thinkable that calls for different parameters have different concrete views
    let result = Type::clone_voidable(function_type.return_type, types);
    return Ok(result);
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
fn calculate_and_store_type<'a>(expr: &'a mut Expression, scopes: &DefinitionScopeStack) -> Result<Ref<'a, Option<Type>>, CompileError> {
    let result = match expr {
        Expression::Call(call) => {
            if call.result_type_cache.borrow().is_some() {
                return Ok(Ref::map(call.result_type_cache.borrow(), |x| &x.unwrap()));
            }
            let mut parameter_types = call.parameters.iter()
                    .map(|p| (calculate_and_store_type_nonvoid(p, scopes), p.pos())).fuse();

            if let Expression::Variable(var) = &call.function {
                if let Identifier::BuiltIn(op) = &var.identifier {
                    *call.result_type_cache.borrow_mut() = Some(
                        Some(calculate_builtin_call_result_type(*op, call.pos(), parameter_types)?)
                    );
                } else {
                    let called_function_type = calculate_and_store_type(&mut call.function, scopes)?.unwrap();
                    *call.result_type_cache.borrow_mut() = Some(
                        calculate_defined_function_call_result_type(call, &called_function_type, parameter_types)?
                    )
                }
            } else {
                let called_function_type = calculate_and_store_type(&mut call.function, scopes)?.unwrap();
                *call.result_type_cache.borrow_mut() = Some(
                    calculate_defined_function_call_result_type(call, &called_function_type, parameter_types)?
                )
            };
            assert!(parameter_types.next().is_none());
            return Ok(Ref::map(call.result_type_cache.borrow(), |x| &x.unwrap()));
        },
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => unimplemented!(),
            Identifier::BuiltIn(op) => panic!("Called calculate_type() on builtin identifier {:?}, but builtin identifiers can have different types depending on context", op)
        },
        Expression::Literal(lit) => unimplemented!()
    };
}

type ConcreteViewComputationResult = Result<(), ()>;

///
/// Calculates the concrete view type of the given builtin function call, and returns it.
/// Expects the concrete view types of the parameters to be stored.
/// Expects its type and the type of all parameters to be already computed and stored.
/// 
fn set_builtin_call_result_concrete_view(call: &FunctionCall, op: BuiltInIdentifier, scopes: &DefinitionScopeStack, type_lifetime: &mut LifetimeMut) -> ConcreteViewComputationResult
{
    assert!(call.function == op);
    assert!(call.result_type_cache.get().is_some());

    let set_result = |concrete: Box<dyn ConcreteView>, lifetime: &mut LifetimeMut| {
        match call.get_stored_type().deref_mut(lifetime) {
            Type::View(view) => view.concrete = Some(concrete),
            _ => panic!("set_result in set_builtin_call_result_concrete_view() called for a non-view type!")
        };
    };

    match op {
        BuiltInIdentifier::ViewZeros => {
            set_result(Box::new(ZeroView::new()), type_lifetime);
            return Ok(());
        },
        BuiltInIdentifier::FunctionIndex => {
            assert!(call.get_stored_type().deref(type_lifetime.as_const()).is_view());
            let indexed_type = get_expression_type(&call.parameters[0], scopes).unwrap();
            let concrete: Box<dyn ConcreteView> = match indexed_type.deref(type_lifetime.as_const()) {
                Type::View(view) => {
                    debug_assert!(view.base.dimension == call.parameters.len() - 1);
                    if let Some(indexed_concrete_view) = &view.concrete {
                        Box::new(ComposedView::compose(Box::new(IndexView::new(view.base.dimension)), indexed_concrete_view.dyn_clone()))
                    } else {
                        return Err(());
                    }
                },
                Type::Array(arr) => {
                    debug_assert!(arr.dimension == call.parameters.len() - 1);
                    Box::new(IndexView::new(arr.dimension))
                },
                _ => unimplemented!()
            };
            set_result(concrete, type_lifetime);
            return Ok(());
        },
        BuiltInIdentifier::FunctionAdd |
            BuiltInIdentifier::FunctionMul |
            BuiltInIdentifier::FunctionAnd |
            BuiltInIdentifier::FunctionOr |
            BuiltInIdentifier::FunctionEq |
            BuiltInIdentifier::FunctionGeq |
            BuiltInIdentifier::FunctionLeq |
            BuiltInIdentifier::FunctionNeq |
            BuiltInIdentifier::FunctionLs |
            BuiltInIdentifier::FunctionGt |
            BuiltInIdentifier::FunctionUnaryDiv | 
            BuiltInIdentifier::FunctionUnaryNeg => 
        {
            return Ok(());
        }
    };
}

///
/// Calculates the concrete view type of the given defined function call, and returns it.
/// Expects the concrete view types of the parameters to be stored.
/// Expects its type and the type of all parameters to be already computed and stored.
/// This will also use the generic view signature of the function (e.g. &Template1, &Template2 -> &Template1), 
/// so this must already be available (work in functions in topological order)
/// 
fn set_defined_call_result_concrete_view(call: &FunctionCall, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> ConcreteViewComputationResult
{
    assert!(call.result_type_cache.get().is_some());

    let result_ptr = call.result_type_cache.get().unwrap();

    if let VoidableTypePtr::Some(ptr) = result_ptr {
        if ptr.deref(types.get_lifetime()).is_code_determined() {
            return Ok(());
        }
        
        let to_replace = {
            let function_type = get_expression_type(&call.function, scopes).unwrap().deref(types.get_lifetime()).expect_callable(call.pos()).internal_error();
            function_type.param_types(types.get_lifetime()).zip(call.parameters.iter()).filter_map(|(param_type, given_param)| {

                let given_param_type = get_expression_type(given_param, scopes).unwrap();

                if let Ok(view) = given_param_type.deref(types.get_lifetime()).expect_view(&TextPosition::NONEXISTING) {
                    assert!(param_type.is_view());
                    if let Some(template) = view.concrete.as_ref().unwrap().downcast::<Template>() {
                        Some((*template, view.concrete.clone().unwrap()))
                    } else {
                        None
                    }
                } else {
                    assert!(!param_type.is_view());
                    None
                }
            }).collect::<Vec<_>>()
        };

        for (template, target) in to_replace {
            Type::replace_templated_view_parts(ptr, template, &*target, &mut types.get_lifetime_mut());
        }
    }
    return Ok(());
}

///
/// Recursively assigns each function call that is nested in the given expression (evtl. including the current expression) a
/// concrete view type, as infered from its parameter, if the return type of the function call is a view type.
/// Requires that the type itself is already correctly calculated and cached.
/// This will also use the generic view signature of called functions (e.g. &Template1, &Template2 -> &Template1), 
/// so this must already be available (work in functions in topological order)
/// 
fn fill_expression_concrete_view(expr: &Expression, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> ConcreteViewComputationResult {
    match expr {
        Expression::Call(call) => {
            for param in &call.parameters {
                fill_expression_concrete_view(param, scopes, types)?;
            }
            match &call.function {
                Expression::Variable(var) => match &var.identifier {
                    Identifier::BuiltIn(op) => {
                        set_builtin_call_result_concrete_view(call, *op, scopes, &mut types.get_lifetime_mut())?;
                    },
                    Identifier::Name(_) => {
                        set_defined_call_result_concrete_view(call, scopes, types)?;
                    }
                },
                _ => {
                    fill_expression_concrete_view(&call.function, scopes, types)?;
                     set_defined_call_result_concrete_view(call, scopes, types)?;
                }
            }
        },
        _ => {}
    };
    return Ok(());
}

fn calculate_and_store_type_nonvoid(
    expr: &Expression, 
    scopes: &DefinitionScopeStack
) -> Result<TypePtr, CompileError> {
    calculate_and_store_type(expr, scopes, types).and_then(|t| t.expect_nonvoid(expr.pos()))
}

fn fill_statement_concrete_view(
    statement: &dyn Statement, 
    scopes: &DefinitionScopeStack, 
    types: &mut TypeVec
) -> Result<(), CompileError> {

    for expr in statement.expressions() {
        fill_expression_concrete_view(expr, &scopes, types).unwrap();
    }

    if let Some(local_var) = statement.downcast::<LocalVariableDeclaration>() {
        if let Some(val) = &local_var.value {
            let assigned_type = get_expression_type(val, scopes).unwrap()
                .deref(types.get_lifetime());
            if let Type::View(view) = assigned_type {
                let concrete_view = view.get_concrete().unwrap().dyn_clone();
                let mut lifetime_mut = types.get_lifetime_mut();
                let var_type = local_var.declaration.variable_type.deref_mut(&mut lifetime_mut);
                var_type.unwrap_view_mut().concrete = Some(concrete_view);
            }
        } else if local_var.declaration.variable_type.deref(types.get_lifetime()).is_view() {
            return Err(
                error_view_not_initialized(&local_var, types.get_lifetime())
            );
        }
    }
    return Ok(());
}

fn determine_types_in_block(
    block: &Block, 
    parent_scopes: &DefinitionScopeStack, 
    types: &mut TypeVec
) -> Result<(), CompileError> {
    let scopes = parent_scopes.child_scope(block);
    for statement in &block.statements {
        for expr in statement.expressions() {
            calculate_and_store_type(expr, &scopes, types)?;
        }
    }
    for statement in &block.statements {
        fill_statement_concrete_view(&**statement, &scopes, types)?;
    }
    for statement in &block.statements {
        for subblock in statement.subblocks() {
            determine_types_in_block(subblock, &scopes, types)?;
        }
    }
    return Ok(());
}

pub fn determine_types_in_function(
    function: &Function,
    global_scope: &DefinitionScopeStack,
    types: &mut TypeVec
) -> Result<(), CompileError> {
    let function_scope = global_scope.child_scope(&*function);
    if let Some(body) = &function.body {
        determine_types_in_block(body, &function_scope, types)?;
    }
    return Ok(());
}

pub fn determine_types_in_program(program: &mut Program) -> Result<(), CompileError> {
    let global_scope = DefinitionScopeStack::new(&program.items[..]);
    for function in &program.items {
        FunctionType::fill_concrete_views_with_template(function.function_type, &mut program.types.get_lifetime_mut());
    }
    for function in call_graph_topological_sort(&program.items)? {
        determine_types_in_function(function, &global_scope, &mut program.types)?;
    }
    return Ok(());
}

pub trait TypeStored {
    fn get_stored_voidable_type(&self) -> VoidableTypePtr;

    fn get_stored_type(&self) -> TypePtr {
        self.get_stored_voidable_type().unwrap()
    }
}

impl TypeStored for FunctionCall {
    fn get_stored_voidable_type(&self) -> VoidableTypePtr {
        self.result_type_cache.get().expect("Called get_type() on a function call expression whose type has not yet been calculated")
    }
}

impl TypeStored for Literal {
    fn get_stored_voidable_type(&self) -> VoidableTypePtr {
        VoidableTypePtr::Some(self.literal_type)
    }
}

pub fn get_expression_type(expr: &Expression, scopes: &DefinitionScopeStack) -> VoidableTypePtr {
    match expr {
        Expression::Call(call) => {
            call.get_stored_voidable_type()
        },
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => VoidableTypePtr::Some(scopes.get_defined(name, var.pos()).internal_error().get_type()),
            Identifier::BuiltIn(op) => panic!("Called calculate_type() on builtin identifier {:?}, but builtin identifiers can have different types depending on context", op)
        },
        Expression::Literal(lit) => lit.get_stored_voidable_type()
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