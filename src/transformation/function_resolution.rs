use super::super::language::prelude::*;

pub type DefinedFunctions = [Box<Function>];

pub enum FunctionDefinition<'a> {
    Builtin(BuiltInIdentifier),
    UserDefined(&'a Function),
}

pub fn find_function_definition<'b>(
    expr: &Expression,
    all_functions: &'b DefinedFunctions,
) -> Result<FunctionDefinition<'b>, CompileError> {
    if let Expression::Variable(identifier) = expr {
        match &identifier.identifier {
            Identifier::Name(name) => {
                let found_function =
                    all_functions
                        .iter()
                        .find(|f| f.identifier == *name)
                        .ok_or(CompileError::new(
                            expr.pos(),
                            format!("Could not find definition of function {}", name),
                            ErrorType::UndefinedSymbol,
                        ));
                Ok(FunctionDefinition::UserDefined(found_function?))
            }
            Identifier::BuiltIn(builtin_identifier) => {
                Ok(FunctionDefinition::Builtin(*builtin_identifier))
            }
        }
    } else {
        panic!("Cannot inline calls to functions that are the result of an expression");
    }
}
