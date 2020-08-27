use super::super::language::prelude::*;
use super::scope::*;
use super::type_error::*;

pub trait Typed {
    fn calculate_type<'a, 'b>(
        &self,
        context: &DefinitionScopeStack<'a, 'b>,
    ) -> Result<Type, CompileError>;
}

impl Typed for Variable {
    fn calculate_type(&self, context: &DefinitionScopeStack) -> Result<Type, CompileError> {

        Ok(context
            .get(&self.identifier.unwrap_name())
            .ok_or_else(|| error_undefined_symbol(&self))?
            .calc_type())
    }
}

impl Typed for Expression {
    fn calculate_type(&self, context: &DefinitionScopeStack) -> Result<Type, CompileError> {

        Ok(match self {
            Expression::Call(call) => {
                match &call.function.expect_identifier().unwrap().identifier {
                    Identifier::Name(name) => {

                        let function = context.get(name).ok_or_else(|| {

                            error_undefined_symbol(&call.function.expect_identifier().unwrap())
                        })?;

                        let return_type = match function.calc_type() {
                            Type::Function(_, return_type) => return_type,
                            ty => Err(error_not_callable(call.function.pos(), &ty))?,
                        };

                        return Ok(*return_type.clone().unwrap());
                    }
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionAdd)
                    | Identifier::BuiltIn(BuiltInIdentifier::FunctionMul)
                    | Identifier::BuiltIn(BuiltInIdentifier::FunctionUnaryDiv)
                    | Identifier::BuiltIn(BuiltInIdentifier::FunctionUnaryNeg) => {
                        call.parameters[0].calculate_type(context)?.without_view()
                    }
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionIndex) => {

                        let array_type = call.parameters[0].calculate_type(context)?;

                        match &array_type {
                            Type::Array(base_type, _) => {
                                Type::View(Box::new(Type::Primitive(*base_type)))
                            }
                            Type::View(viewn) => match &**viewn {
                                Type::Array(base_type, _) => {
                                    Type::View(Box::new(Type::Primitive(*base_type)))
                                }
                                _ => Err(error_not_indexable(call.pos(), &array_type))?,
                            },
                            _ => Err(error_not_indexable(call.pos(), &array_type))?,
                        }
                    }
                    _ => unimplemented!(),
                }
            }
            Expression::Literal(_) => Type::Primitive(PrimitiveType::Int),
            Expression::Variable(var) => var.calculate_type(context)?,
        })
    }
}
