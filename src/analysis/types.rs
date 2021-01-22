use super::super::language::prelude::*;
use super::super::language::position::NONEXISTING;
use super::scope::*;
use super::topological_sort::call_graph_topological_sort;
use super::type_error::*;

fn parent_type(a: PrimitiveType, b: PrimitiveType) -> PrimitiveType
{
    match (a, b) {
        (PrimitiveType::Int, PrimitiveType::Int) => PrimitiveType::Int,
        _ => PrimitiveType::Float
    }
}

///
/// Calculates the result type of the builtin operator applied to parameters of the given types.
/// 
/// If the result type is a view, this might create new type objects in the type vector. To prevent
/// creating the same view type object for the same expression multiple times, the caller should cache
/// the result in the corresponding function call ast node.
/// 
/// This also performs type checking, i.e. will return a compile error if the types do not match within the expression.
/// 
/// Created view types have no concrete type information set.
/// 
fn calculate_builtin_call_result_type<'a, I>(op: BuiltInIdentifier, pos: &TextPosition, types: &mut TypeVec, mut param_types: I) -> Result<TypePtr, CompileError>
    where I: Iterator, I::Item: FnOnce(&mut TypeVec) -> (Result<TypePtr, CompileError>, &'a TextPosition)
{
    match op {
        BuiltInIdentifier::ViewZeros => {
            let mut dimension_count = 0;
            for param in param_types {
                let (param_type_try, pos) = param(types);
                let param_type = param_type_try?;
                if param_type.deref(types.get_lifetime()).expect_scalar(pos)? != PrimitiveType::Int {
                    return Err(error_type_not_convertable(pos, param_type, types.get_primitive_type(PrimitiveType::Int, false), types.get_lifetime()));
                }
                dimension_count += 1;
            }
            return Ok(types.get_generic_view_type(PrimitiveType::Int, dimension_count, false));
        },
        BuiltInIdentifier::FunctionAdd |
            BuiltInIdentifier::FunctionMul => 
        {
            let base_type = param_types.try_fold(PrimitiveType::Int, |current, next_param| {
                let (ty, pos) = next_param(types);
                let next_primitive_type = ty?.deref(types.get_lifetime()).expect_arithmetic(pos)?;
                Ok(parent_type(current, next_primitive_type))
            })?;
            return Ok(types.get_array_type(base_type, 0, false));
        },
        BuiltInIdentifier::FunctionAnd |
            BuiltInIdentifier::FunctionOr => 
        {
            for param in param_types {
                let (param_type_try, pos) = param(types);
                let param_type = param_type_try?;
                if param_type.deref(types.get_lifetime()).expect_scalar(pos)? != PrimitiveType::Bool {
                    return Err(error_type_not_convertable(pos, param_type, types.get_primitive_type(PrimitiveType::Bool, false), types.get_lifetime()));
                }
            }
            return Ok(types.get_array_type(PrimitiveType::Bool, 0, false));
        },
        BuiltInIdentifier::FunctionEq |
            BuiltInIdentifier::FunctionGeq |
            BuiltInIdentifier::FunctionLeq |
            BuiltInIdentifier::FunctionNeq |
            BuiltInIdentifier::FunctionLs |
            BuiltInIdentifier::FunctionGt => 
        {
            for param in param_types {
                let (param_type_try, pos) = param(types);
                let param_type = param_type_try?;
                param_type.deref(types.get_lifetime()).expect_arithmetic(pos)?;
            }
            return Ok(types.get_array_type(PrimitiveType::Bool, 0, false));
        },
        BuiltInIdentifier::FunctionIndex => {
            let (array_type, _pos) = param_types.next().expect("index function call has no parameters, but the indexed array should be the first parameter")(types);
            for param in param_types {
                let (param_type_try, pos) = param(types);
                let param_type = param_type_try?;
                if param_type.deref(types.get_lifetime()).expect_scalar(pos)? != PrimitiveType::Int {
                    return Err(error_type_not_convertable(pos, param_type, types.get_primitive_type(PrimitiveType::Int, false), types.get_lifetime()));
                }
            }
            let indexed_type = types.get_lifetime().cast(array_type?).clone();
            return match indexed_type {
                Type::View(view) => Ok(types.get_generic_view_type(
                    view.base.base, 
                    0, 
                    view.base.mutable
                )),
                Type::Array(arr) => Ok(types.get_generic_view_type(
                    arr.base, 
                    0, 
                    arr.mutable, 
                )),
                _ => unimplemented!()
            };
        },
        BuiltInIdentifier::FunctionUnaryDiv | BuiltInIdentifier::FunctionUnaryNeg => {
            let (result_try, pos) = param_types.next().ok_or_else(|| error_wrong_parameter_count(pos, Identifier::BuiltIn(op), 1))?(types);
            let result = result_try?;
            result.deref(types.get_lifetime()).expect_arithmetic(pos)?;
            return Ok(result);
        }
    }
}

///
/// Calculates the result type of the given expression, including correct concrete view types. If the
/// expression is a function call expression, the calculated type pointer will also be written into 
/// the type cache of that node. The parameter types of a function call will always be computed and
/// stored in the corresponding cache.
/// 
/// This will not set concrete view types.
/// 
/// This also performs type checking, i.e. will return a compile error if the types do not match within the expression.
/// 
fn calculate_and_store_type<'a>(expr: &'a Expression, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> Result<VoidableTypePtr, CompileError> {
    let result = match expr {
        Expression::Call(call) => {
            let call_type = if let Some(ty) = call.result_type_cache.get() {
                ty
            } else if let Expression::Variable(var) = &call.function {
                let parameter_types = call.parameters.iter()
                        .map(|p| move |types: &mut TypeVec| (calculate_and_store_type_nonvoid(p, scopes, types), p.pos()));
                if let Identifier::BuiltIn(op) = &var.identifier {
                    VoidableTypePtr::Some(calculate_builtin_call_result_type(*op, call.pos(), types, parameter_types)?)
                } else {
                    calculate_defined_function_call_result_type(call, calculate_and_store_type(&call.function, scopes, types)?.unwrap(), scopes, types)?
                }
            } else {
                calculate_defined_function_call_result_type(call, calculate_and_store_type(&call.function, scopes, types)?.unwrap(), scopes, types)?
            };
            call.result_type_cache.set(Some(call_type));
            Ok(call_type)
        },
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => Ok(VoidableTypePtr::Some(scopes.get_defined(name, var.pos())?.get_type())),
            Identifier::BuiltIn(op) => panic!("Called calculate_type() on builtin identifier {}, but builtin identifiers can have different types depending on context", op)
        },
        Expression::Literal(lit) => Ok(VoidableTypePtr::Some(lit.literal_type))
    };
    return result;
}

///
/// This also performs type checking, i.e. will return a compile error if the types do not match within the expression.
/// 
/// This will not explicitly set concrete view types, but copy them with the function result type.
///
fn calculate_defined_function_call_result_type(call: &FunctionCall, function_type_ptr: TypePtr, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> Result<VoidableTypePtr, CompileError> {
    let param_types = call.parameters.iter().map(|p| (calculate_and_store_type_nonvoid(p, scopes, types), p.pos())).collect::<Vec<_>>();
    let function_type = types.get_lifetime().cast(function_type_ptr).expect_callable(call.pos()).internal_error();
    for (given_param, expected_param) in param_types.into_iter().zip(function_type.param_types.iter()) {
        let param_type = given_param.0?;
        if !param_type.deref(types.get_lifetime()).is_implicitly_convertable(expected_param.deref(types.get_lifetime()), types.get_lifetime()) {
            return Err(error_type_not_convertable(given_param.1, param_type, *expected_param, types.get_lifetime()));
        }
    }
    debug_assert!(function_type.return_type.is_void() || !function_type.return_type.unwrap().deref(types.get_lifetime()).is_view());

    // clone the type here, as it is thinkable that calls for different parameters have different concrete views
    let result = Type::clone_voidable(function_type.return_type, types);
    return Ok(result);
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

                if let Ok(view) = given_param_type.deref(types.get_lifetime()).expect_view(&NONEXISTING) {
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
fn set_expression_concrete_view(expr: &Expression, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> ConcreteViewComputationResult {
    match expr {
        Expression::Call(call) => {
            for param in &call.parameters {
                set_expression_concrete_view(param, scopes, types)?;
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
                    set_expression_concrete_view(&call.function, scopes, types)?;
                     set_defined_call_result_concrete_view(call, scopes, types)?;
                }
            }
        },
        _ => {}
    };
    return Ok(());
}

fn calculate_and_store_type_nonvoid<'a>(expr: &'a Expression, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> Result<TypePtr, CompileError> {
    calculate_and_store_type(expr, scopes, types).and_then(|t| t.expect_nonvoid(expr.pos()))
}

fn determine_types_in_block<'a>(block: &'a Block, parent_scopes: &DefinitionScopeStack, types: &mut TypeVec) -> Result<(), CompileError> {
    let scopes = parent_scopes.child_scope(block);
    for statement in &block.statements {
        for expr in statement.expressions() {
            calculate_and_store_type(expr, &scopes, types)?;
        }
    }
    for statement in &block.statements {
        for expr in statement.expressions() {
            set_expression_concrete_view(expr, &scopes, types).unwrap();
        }
    }
    for statement in &block.statements {
        for subblock in statement.subblocks() {
            determine_types_in_block(subblock, &scopes, types)?;
        }
    }
    return Ok(());
}

pub fn determine_types_in_program(program: &mut Program) -> Result<(), CompileError> {
    let global_scope = DefinitionScopeStack::new(&program.items[..]);
    for function in &program.items {
        FunctionType::fill_concrete_views_with_template(function.function_type, &mut program.types.get_lifetime_mut());
    }
    for function in call_graph_topological_sort(&program.items)? {
        let function_scope = global_scope.child_scope(&*function);
        if let Some(body) = &function.body {
            determine_types_in_block(body, &function_scope, &mut program.types)?;
        }
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
            Identifier::BuiltIn(op) => panic!("Called calculate_type() on builtin identifier {}, but builtin identifiers can have different types depending on context", op)
        },
        Expression::Literal(lit) => lit.get_stored_voidable_type()
    }
}