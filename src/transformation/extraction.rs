use super::super::language::prelude::*;
use super::function_resolution::{find_function_definition, DefinedFunctions, FunctionDefinition};

pub struct Extractor<F>
where
    F: FnMut(&FunctionCall, &Function) -> bool,
{
    pub should_extract: F,
}

impl<F> Extractor<F>
where
    F: FnMut(&FunctionCall, &Function) -> bool,
{
    ///
    /// Returns a new local variable and inserts statements that initialize this variable with the result value of the
    /// given call to the given function into `previous_declaration_statements`. This is done recursivly on all parameters.
    /// If the call does not yield a result, only its parameters are modified (in this case, it is already "extracted")
    ///
    fn extract_function_call<'a, 'b, G>(
        &mut self,
        function_call: FunctionCall,
        function_definition: &'b Function,
        rename_disjunct: &'a mut G,
        defined_functions: &'b DefinedFunctions,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
    ) -> Expression
    where
        G: FnMut(Name) -> Name,
        'b: 'a,
    {
        let pos = function_call.pos().clone();
        let recursive_extract = |expr: Expression| {
            self.extract_calls_in_expr(
                expr,
                rename_disjunct,
                defined_functions,
                previous_declaration_statements,
            )
        };
        if let Some(return_type) = &function_definition.return_type {
            let variable_name = (*rename_disjunct)(Name::new(
                format!("result_{}", function_definition.identifier.name),
                0,
            ));
            let declaration = LocalVariableDeclaration {
                declaration: Declaration {
                    pos: pos.clone(),
                    variable: variable_name.clone(),
                    variable_type: return_type.clone(),
                },
                value: Some(Expression::Call(Box::new(
                    self.extract_calls_in_parameters(
                        function_call,
                        rename_disjunct,
                        defined_functions,
                        previous_declaration_statements,
                    ),
                ))),
            };
            previous_declaration_statements.push(Box::new(declaration));
            return Expression::Variable(Variable {
                pos: pos,
                identifier: Identifier::Name(variable_name),
            });
        } else {
            return Expression::Call(Box::new(self.extract_calls_in_parameters(
                function_call,
                rename_disjunct,
                defined_functions,
                previous_declaration_statements,
            )));
        }
    }

    fn extract_calls_in_parameters<'a, 'b, G>(
        &mut self,
        mut call: FunctionCall,
        rename_disjunct: &'a mut G,
        defined_functions: &'b DefinedFunctions,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
    ) -> FunctionCall
    where
        G: FnMut(Name) -> Name,
        'b: 'a,
    {
        let recursive_extract = |expr: Expression| {
            self.extract_calls_in_expr(
                expr,
                rename_disjunct,
                defined_functions,
                previous_declaration_statements,
            )
        };
        call.parameters = call.parameters.into_iter().map(recursive_extract).collect();
        return call;
    }

    pub fn extract_calls_in_expr<'a, 'b, G>(
        &mut self,
        expression: Expression,
        rename_disjunct: &'a mut G,
        defined_functions: &'b DefinedFunctions,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
    ) -> Expression
    where
        G: FnMut(Name) -> Name,
        'b: 'a,
    {
        match expression {
            Expression::Call(call) => {
                match find_function_definition(&call.function, defined_functions).internal_error() {
                    FunctionDefinition::UserDefined(definition)
                        if (self.should_extract)(&call, definition) =>
                    {
                        self.extract_function_call(
                            *call,
                            definition,
                            rename_disjunct,
                            defined_functions,
                            previous_declaration_statements,
                        )
                    }
                    FunctionDefinition::UserDefined(_) => {
                        Expression::Call(Box::new(self.extract_calls_in_parameters(
                            *call,
                            rename_disjunct,
                            defined_functions,
                            previous_declaration_statements,
                        )))
                    }
                    _ => Expression::Call(Box::new(self.extract_calls_in_parameters(
                        *call,
                        rename_disjunct,
                        defined_functions,
                        previous_declaration_statements,
                    ))),
                }
            }
            expr => expr,
        }
    }
}
