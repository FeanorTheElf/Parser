use super::super::language::prelude::*;
use super::symbol::*;
use super::scope::*;

pub trait Typed {
    fn calculate_type(&self, context: &ScopeStack<&dyn SymbolDefinition>) -> Type;
}

impl Typed for Name {
    fn calculate_type(&self, context: &ScopeStack<&dyn SymbolDefinition>) -> Type {
        context.get(self).expect(format!("Expect {} to be defined", self).as_str()).calc_type()
    }
}

impl Typed for Expression {
    fn calculate_type(&self, context: &ScopeStack<&dyn SymbolDefinition>) -> Type {
        match self {
            Expression::Call(call) => match &call.function.expect_identifier().unwrap() {
                Identifier::Name(name) => context.get(name).unwrap().dynamic().downcast_ref::<Function>().unwrap().return_type.as_ref().unwrap().clone(),
                Identifier::BuiltIn(BuiltInIdentifier::FunctionAdd)
                    | Identifier::BuiltIn(BuiltInIdentifier::FunctionMul) 
                    | Identifier::BuiltIn(BuiltInIdentifier::FunctionUnaryDiv) 
                    | Identifier::BuiltIn(BuiltInIdentifier::FunctionUnaryNeg) => call.parameters[0].calculate_type(context),
                Identifier::BuiltIn(BuiltInIdentifier::FunctionIndex) => {
                    let array_type = call.parameters[0].calculate_type(context);
                    match array_type {
                        Type::Array(base_type, _) => Type::View(Box::new(Type::Primitive(base_type))),
                        _ => CompileError::new(call.pos(), format!("Cannot index access on {}", array_type), ErrorType::TypeError).throw()
                    }
                }
                _ => unimplemented!()
            },
            Expression::Literal(lit) => Type::Primitive(PrimitiveType::Int),
            Expression::Variable(var) => match &var.identifier {
                Identifier::Name(name) => name.calculate_type(context),
                Identifier::BuiltIn(_) => unimplemented!()
            }
        }
    }
}