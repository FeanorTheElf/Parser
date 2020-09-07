use super::super::language::prelude::*;
use super::scope::*;
use super::type_error::*;

use super::super::util::dyn_lifetime::*;

pub trait Typed {
    fn calculate_type<'a, 'b>(
        &self,
        context: &DefinitionScopeStack<'a, 'b>,
        prog_lifetime: Lifetime
    ) -> Result<Type, CompileError>;
}

impl Typed for Variable {
    fn calculate_type(&self, context: &DefinitionScopeStack, prog_lifetime: Lifetime) -> Result<Type, CompileError> {

        Ok(context
            .get(&self.identifier.unwrap_name())
            .ok_or_else(|| error_undefined_symbol(&self))?
            .calc_type(prog_lifetime))
    }
}

impl Typed for Expression {
    fn calculate_type(&self, context: &DefinitionScopeStack, prog_lifetime: Lifetime) -> Result<Type, CompileError> {

        Ok(match self {
            Expression::Call(call) => {
                match &call.function.expect_identifier().unwrap().identifier {
                    Identifier::Name(name) => {

                        let function = context.get(name).ok_or_else(|| {

                            error_undefined_symbol(&call.function.expect_identifier().unwrap())
                        })?;

                        let return_type = function.calc_type(prog_lifetime).expect_callable(call.function.pos())?
                            .return_type.map(|t| prog_lifetime.cast(t).borrow().clone());

                        return Ok(return_type.clone().unwrap());
                    }
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionAdd)
                    | Identifier::BuiltIn(BuiltInIdentifier::FunctionMul)
                    | Identifier::BuiltIn(BuiltInIdentifier::FunctionUnaryDiv)
                    | Identifier::BuiltIn(BuiltInIdentifier::FunctionUnaryNeg) => {
                        call.parameters[0].calculate_type(context, prog_lifetime)?.without_view()
                    }
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionIndex) => {

                        let array_type = call.parameters[0].calculate_type(context, prog_lifetime)?;

                        Type::View(ViewType { base: ArrayType { base: array_type.expect_indexable(call.pos())?.base, dimension: 0 }, concrete: None })
                    }
                    _ => unimplemented!(),
                }
            }
            Expression::Literal(_) => Type::Array(ArrayType { base: PrimitiveType::Int, dimension: 0 }),
            Expression::Variable(var) => var.calculate_type(context, prog_lifetime)?,
        })
    }
}
