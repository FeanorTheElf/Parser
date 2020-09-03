use super::super::analysis::scope::NameScopeStack;
use super::super::language::prelude::*;
use super::function_resolution::{find_function_definition, DefinedFunctions, FunctionDefinition};

pub struct Extractor<F>
where
    F: FnMut(&FunctionCall, &Function) -> bool,
{
    should_extract: F,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExtractionReport {
    pub extracted_var_declaration_index: usize,
    pub extracted_var_value_assignment_index: usize,
}

impl std::cmp::PartialOrd for ExtractionReport {
    fn partial_cmp(&self, rhs: &ExtractionReport) -> Option<std::cmp::Ordering> {
        Some(self.cmp(rhs))
    }
}

impl std::cmp::Ord for ExtractionReport {
    fn cmp(&self, rhs: &ExtractionReport) -> std::cmp::Ordering {
        match self.extracted_var_value_assignment_index.cmp(&rhs.extracted_var_value_assignment_index) {
            std::cmp::Ordering::Equal => self.extracted_var_declaration_index.cmp(&rhs.extracted_var_declaration_index),
            ord => ord
        }
    }
}

impl<F> Extractor<F>
where
    F: FnMut(&FunctionCall, &Function) -> bool,
{
    pub fn new(should_extract: F) -> Self {

        Extractor {
            should_extract: should_extract,
        }
    }

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

            self.extract_expression_recursive(
                expr,
                rename_disjunct,
                defined_functions,
                previous_declaration_statements,
            )
        };

        call.parameters = call.parameters.into_iter().map(recursive_extract).collect();

        return call;
    }

    pub fn extract_expression_recursive<'a, 'b, G>(
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

    pub fn extract_calls_in_block_flat<'a, 'b>(
        &mut self,
        block: &'a mut Block,
        parent_scopes: &'b NameScopeStack<'b>,
        defined_functions: &'b DefinedFunctions,
    ) -> Vec<ExtractionReport> {

        let scopes = parent_scopes.child_scope(block);

        let mut results = Vec::new();

        let mut result_statements: Vec<Box<dyn Statement>> = Vec::new();

        let mut rename_disjunct = scopes.rename_disjunct();

        for mut statement in block.statements.drain(..) {

            let index_before = result_statements.len();

            for expression in statement.iter_mut() {

                take_mut::take(expression, &mut |expr| {

                    self.extract_expression_recursive(
                        expr,
                        &mut rename_disjunct,
                        defined_functions,
                        &mut result_statements,
                    )
                });
            }

            let index_after = result_statements.len();

            // already allocate entries for the initialization of the extracted result variables
            for i in 0..(index_after - index_before) {

                let init_block = Block {
                    pos: position::NONEXISTING,
                    statements: Vec::new(),
                };

                result_statements.push(Box::new(init_block));

                results.push(ExtractionReport {
                    extracted_var_declaration_index: index_before + i,
                    extracted_var_value_assignment_index: index_after + i,
                });
            }

            result_statements.push(statement);
        }

        block.statements = result_statements;

        return results;
    }

    pub fn extract_calls_in_block<'a, 'b>(
        &mut self,
        block: &'a mut Block,
        parent_scopes: &'b NameScopeStack<'b>,
        defined_functions: &'b DefinedFunctions,
    ) {
        let mut extractions = self.extract_calls_in_block_flat(block, parent_scopes, defined_functions);
        extractions.sort();
        extractions.reverse();
        for extraction in extractions {
            debug_assert!(&*block.statements[extraction.extracted_var_value_assignment_index] == &Block {
                pos: position::NONEXISTING,
                statements: vec![]
            } as &dyn Statement);
            block.statements.remove(extraction.extracted_var_value_assignment_index);
        }
        let child_scopes = parent_scopes.child_scope(block);
        for statement in &mut block.statements {
            for mut subblock in statement.iter_mut() {
                self.extract_calls_in_block(&mut subblock, &child_scopes, defined_functions);
            }
        }
    }

    pub fn extract_calls_in_program(&mut self, program: &mut Program) {

        assert_ne!(program.items.len(), 0);

        let scopes = NameScopeStack::new(&program.items[..]);

        for i in 0..program.items.len() {

            program.items.swap(0, i);

            let (current, other) = program.items[..].split_at_mut(1);

            let child_scopes = scopes.child_scope(&*current[0]);

            if let Some(body) = &mut current[0].body {

                self.extract_calls_in_block(body, &child_scopes, other);
            }
        }
    }
}

#[cfg(test)]
use super::super::lexer::lexer::lex_str;
#[cfg(test)]
use super::super::parser::Parser;

#[test]
fn test_extract_calls_in_program() {
    let mut program = Program::parse(&mut lex_str("
    
    fn foo(a: int,): int native;
    fn bar(a: int,): int native;

    fn main(): int {
        if (foo(bar(a,),)) {
            return bar(foo(a,),);
        }
    }
    ")).unwrap();

    Extractor::new(|_, _| true).extract_calls_in_program(&mut program);

    let expected = Program::parse(&mut lex_str("

    fn foo(a: int,): int native;
    fn bar(a: int,): int native;

    fn main(): int {
        let result_bar: int = bar(a,);
        let result_foo: int = foo(result_bar,);
        if (result_bar) {
            let result_foo#1: int = foo(a,);
            let result_bar#1: int = bar(result_foo#1,);
            return result_bar#1;
        }
    }
    ")).unwrap();

    assert_ast_eq!(expected, program);
}