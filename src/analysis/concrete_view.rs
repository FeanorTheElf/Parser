use super::super::language::prelude::*;
use super::scope::DefinitionScopeStack;

fn parent_type(a: PrimitiveType, b: PrimitiveType) -> PrimitiveType
{
    match (a, b) {
        (PrimitiveType::Int, PrimitiveType::Int) => PrimitiveType::Int,
        _ => PrimitiveType::Float
    }
}

fn calculate_builtin_call_result_type<'a, I>(op: BuiltInIdentifier, mut param_types: I, prog_lifetime: Lifetime) -> Type 
    where I: Iterator<Item = (Type, &'a TextPosition)>
{
    let primitive = |p: PrimitiveType| Type::Array(ArrayType {
        base: p,
        dimension: 0,
        mutable: false
    });
    match op {
        BuiltInIdentifier::ViewZeros => {
            let dimension_count = param_types.count();
            Type::View(ViewType {
                base: ArrayType {
                    base: PrimitiveType::Int,
                    dimension: dimension_count,
                    mutable: false
                },
                concrete: Some(Box::new(ZeroView::new()))
            })
        },
        BuiltInIdentifier::FunctionAdd => {
            primitive(param_types.map(|(t, pos)| t.expect_array(pos).internal_error().base).fold(PrimitiveType::Int, &parent_type))
        },
        BuiltInIdentifier::FunctionAnd |
            BuiltInIdentifier::FunctionOr |
            BuiltInIdentifier::FunctionEq |
            BuiltInIdentifier::FunctionGeq |
            BuiltInIdentifier::FunctionLeq |
            BuiltInIdentifier::FunctionNeq |
            BuiltInIdentifier::FunctionLs |
            BuiltInIdentifier::FunctionGt => primitive(PrimitiveType::Int),
        BuiltInIdentifier::FunctionIndex => {
            let (array_type, pos) = param_types.next().unwrap();
            match array_type {
                _ => unimplemented!()
            };
            unimplemented!()
        },
        _ => unimplemented!()
    }
}

fn calculate_type(expr: &Expression, scopes: &DefinitionScopeStack, prog_lifetime: Lifetime) -> Type {
    match expr {
        Expression::Call(call) => match &call.function {
            Expression::Variable(var) if var.identifier.is_builtin() => calculate_builtin_call_result_type(
                *var.identifier.unwrap_builtin(), 
                call.parameters.iter().map(|p| (calculate_type(p, scopes, prog_lifetime), p.pos())), 
                prog_lifetime
            ),
            func => calculate_type(func, scopes, prog_lifetime).expect_callable(func.pos()).internal_error().return_type(prog_lifetime).unwrap().clone()
        },
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => prog_lifetime.cast(scopes.get_defined(name, var.pos()).internal_error().get_type()).borrow().clone(),
            Identifier::BuiltIn(op) => panic!("Called calculate_type() on builtin identifier {}, but builtin identifiers can have different types depending on context", op)
        },
        Expression::Literal(_) => Type::Array(ArrayType {
            base: PrimitiveType::Int,
            dimension: 0,
            mutable: false
        })
    }
}

fn fill_concrete_view_declaration(decl: &LocalVariableDeclaration, scopes: &DefinitionScopeStack, prog_lifetime: Lifetime) {
    let mut type_ptr = prog_lifetime.cast(decl.declaration.variable_type).borrow_mut();
    if let Type::View(view) = &mut *type_ptr {

    }
}

pub fn fill_concrete_views(func: &Function) {
    
}