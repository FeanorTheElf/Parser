use super::super::language::prelude::*;
use super::scope::*;
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
/// Created view types have no concrete type information set.
/// 
fn calculate_builtin_call_result_type<'a, I>(op: BuiltInIdentifier, pos: &TextPosition, types: &mut TypeVec, mut param_types: I) -> Result<TypePtr, CompileError>
    where I: Iterator, I::Item: FnOnce(&mut TypeVec) -> (Result<TypePtr, CompileError>, &'a TextPosition)
{
    match op {
        BuiltInIdentifier::ViewZeros => {
            let dimension_count = param_types.count();
            Ok(types.get_generic_view_type(PrimitiveType::Int, dimension_count, false))
        },
        BuiltInIdentifier::FunctionAdd |
            BuiltInIdentifier::FunctionMul => 
        {
            let base_type = param_types.try_fold(PrimitiveType::Int, |current, next_param| {
                let (ty, pos) = next_param(types);
                let next_primitive_type = ty?.deref(types.get_lifetime()).expect_array(pos).internal_error().base;
                Ok(parent_type(current, next_primitive_type))
            })?;
            Ok(types.get_array_type(base_type, 0, false))
        },
        BuiltInIdentifier::FunctionAnd |
            BuiltInIdentifier::FunctionOr |
            BuiltInIdentifier::FunctionEq |
            BuiltInIdentifier::FunctionGeq |
            BuiltInIdentifier::FunctionLeq |
            BuiltInIdentifier::FunctionNeq |
            BuiltInIdentifier::FunctionLs |
            BuiltInIdentifier::FunctionGt => Ok(types.get_array_type(PrimitiveType::Int, 0, false)),
        BuiltInIdentifier::FunctionIndex => {
            let (array_type, _pos) = param_types.next().expect("index function call has no parameters")(types);
            match types.get_lifetime().cast(array_type?).clone() {
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
            }
        },
        BuiltInIdentifier::FunctionUnaryDiv | BuiltInIdentifier::FunctionUnaryNeg => {
            let result = param_types.next().ok_or_else(|| error_wrong_parameter_count(pos, Identifier::BuiltIn(op), 1))?(types).0?;
            Ok(result)
        }
    }
}

fn calculate_defined_function_call_result_type(call: &FunctionCall, function_type_ptr: TypePtr, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> Result<VoidableTypePtr, CompileError> {
    let function_type = types.get_lifetime().cast(function_type_ptr).expect_callable(call.pos()).internal_error();
    debug_assert!(function_type.return_type.is_void() || !function_type.return_type.unwrap().deref(types.get_lifetime()).is_view());
    return Ok(function_type.return_type);
}

///
/// Calculates the result type of the given expression, including correct concrete view types. If the
/// expression is a function call expression, the calculated type pointer will also be written into 
/// the type cache of that node. If the parameter types of a function call must be known to calculate
/// the type, they are recursivly calculated.
/// 
/// If some concrete view types that are required to determine the concrete view type of the result
/// are not set, the result will also not have a concrete view type set.
/// 
pub fn calculate_and_store_type<'a>(expr: &'a Expression, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> Result<VoidableTypePtr, CompileError> {
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

pub fn calculate_and_store_type_nonvoid<'a>(expr: &'a Expression, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> Result<TypePtr, CompileError> {
    calculate_and_store_type(expr, scopes, types).and_then(|t| t.expect_nonvoid(expr.pos()))
}

pub fn determine_types_in_statement<'a>(statement: &'a dyn Statement, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> Result<(), CompileError> {
    for expr in statement.expressions() {
        calculate_and_store_type(expr, &scopes, types)?;
    }
    for subblock in statement.subblocks() {
        determine_types_in_block(subblock, &scopes, types)?;
    }
    return Ok(());
}

pub fn determine_types_in_block<'a>(block: &'a Block, parent_scopes: &DefinitionScopeStack, types: &mut TypeVec) -> Result<(), CompileError> {
    let scopes = parent_scopes.child_scope(block);
    for statement in &block.statements {
        determine_types_in_statement(&**statement, &scopes, types)?;
    }
    return Ok(());
}

pub fn determine_types_in_program(program: &mut Program) -> Result<(), CompileError> {
    let global_scope = DefinitionScopeStack::new(&program.items[..]);
    for function in &program.items {
        let function_scope = global_scope.child_scope(&**function);
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