use super::super::language::prelude::*;
use super::scope::DefinitionScopeStack;

fn parent_type(a: PrimitiveType, b: PrimitiveType) -> PrimitiveType
{
    match (a, b) {
        (PrimitiveType::Int, PrimitiveType::Int) => PrimitiveType::Int,
        _ => PrimitiveType::Float
    }
}

fn calculate_builtin_call_result_type<'a, I>(op: BuiltInIdentifier, types: &mut TypeVec, mut param_types: I) -> TypePtr
    where I: Iterator, I::Item: FnOnce(&mut TypeVec) -> (TypePtr, &'a TextPosition)
{
    match op {
        BuiltInIdentifier::ViewZeros => {
            let dimension_count = param_types.count();
            types.get_view_type(PrimitiveType::Int, dimension_count, false, Box::new(ZeroView::new()))
        },
        BuiltInIdentifier::FunctionAdd |
            BuiltInIdentifier::FunctionMul => 
        {
            let base_type = param_types.map(|param| {
                let (ty, pos) = param(types);
                types.get_lifetime().cast(ty).expect_array(pos).internal_error().base
            }).fold(PrimitiveType::Int, &parent_type);
            types.get_array_type(base_type, 0, false)
        },
        BuiltInIdentifier::FunctionAnd |
            BuiltInIdentifier::FunctionOr |
            BuiltInIdentifier::FunctionEq |
            BuiltInIdentifier::FunctionGeq |
            BuiltInIdentifier::FunctionLeq |
            BuiltInIdentifier::FunctionNeq |
            BuiltInIdentifier::FunctionLs |
            BuiltInIdentifier::FunctionGt => types.get_array_type(PrimitiveType::Int, 0, false),
        BuiltInIdentifier::FunctionIndex => {
            let (array_type, _pos) = param_types.next().unwrap()(types);
            match types.get_lifetime().cast(array_type).clone() {
                Type::View(view) => types.get_view_type(
                    view.base.base, 
                    0, 
                    view.base.mutable, 
                    Box::new(ComposedView::compose(Box::new(CompleteIndexView::new()), view.concrete.unwrap()))
                ),
                Type::Array(arr) => types.get_view_type(
                    arr.base, 
                    0, 
                    arr.mutable, 
                    Box::new(CompleteIndexView::new())
                ),
                _ => unimplemented!()
            }
        },
        BuiltInIdentifier::FunctionUnaryDiv |
            BuiltInIdentifier::FunctionUnaryNeg => param_types.next().unwrap()(types).0
    }
}

fn calculate_type(expr: &Expression, scopes: &DefinitionScopeStack, types: &mut TypeVec) -> TypePtr {
    match expr {
        Expression::Call(call) => {
            if let Some(ty) = call.result_type.get() {
                return ty;
            } else if let Expression::Variable(var) = &call.function {
                if let Identifier::BuiltIn(op) = &var.identifier {
                    let parameter_types = call.parameters.iter().map(|p| move |types: &mut TypeVec| (calculate_type(p, scopes, types), p.pos()));
                    let ty = calculate_builtin_call_result_type(*op, types, parameter_types);
                    call.result_type.set(Some(ty));
                    return ty;
                }
            }
            let function_type_ptr = calculate_type(&call.function, scopes, types);
            let function_type = types.get_lifetime().cast(function_type_ptr).expect_callable(call.pos()).internal_error();
            let result_type = *function_type.return_type.as_ref().unwrap();
            debug_assert!(!types.get_lifetime().cast(result_type).is_view());
            return result_type;
        },
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => scopes.get_defined(name, var.pos()).internal_error().get_type(),
            Identifier::BuiltIn(op) => panic!("Called calculate_type() on builtin identifier {}, but builtin identifiers can have different types depending on context", op)
        },
        Expression::Literal(_) => types.get_array_type(PrimitiveType::Int, 0, false)
    }
}

pub fn fill_concrete_views(func: &Function, types: &mut TypeVec) {
    
}