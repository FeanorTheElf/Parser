use super::super::analysis::scope::NameScopeStack;
use super::super::language::prelude::*;
use super::function_resolution::{find_function_definition, DefinedFunctions, FunctionDefinition};
use super::renaming::*;
use std::collections::HashSet;

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
    /// Parameters:
    /// - `function_call`: Call to extract 
    /// - `function_definition`: Definition of the called function
    /// - `rename_disjunct`: Closure that can rename identifiers in a way to be disjunct to all identifiers further up in the scope stack
    /// - `previous_declaration_statements`: Vector to add all initialization statements to
    /// - `prog_lifetime`: Dynamic lifetime description of the program data (as the type vector)+
    /// - `new_definitions`: Insert here all names of newly created local variables  
    ///

    fn extract_function_call<'a, 'b, G>(
        &mut self,
        function_call: FunctionCall,
        function_definition: &'b Function,
        rename_disjunct: &'a mut G,
        defined_functions: &'b DefinedFunctions,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
        prog_lifetime: Lifetime,
        new_definitions: &mut HashSet<Name>
    ) -> Expression
    where
        G: FnMut(Name) -> Name,
        'b: 'a,
    {

        let pos = function_call.pos().clone();

        if let VoidableTypePtr::Some(return_type) = &function_definition.get_type(prog_lifetime).return_type {

            let variable_name = (*rename_disjunct)(Name::new(
                format!("result_{}", function_definition.identifier.name),
                0,
            ));
            new_definitions.insert(variable_name.clone());

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
                        prog_lifetime,
                        new_definitions
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
                prog_lifetime,
                new_definitions
            )));
        }
    }

    fn extract_calls_in_parameters<'a, 'b, G>(
        &mut self,
        mut call: FunctionCall,
        rename_disjunct: &'a mut G,
        defined_functions: &'b DefinedFunctions,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
        prog_lifetime: Lifetime,
        new_definitions: &mut HashSet<Name>
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
                prog_lifetime,
                new_definitions
            )
        };

        call.parameters = call.parameters.into_iter().map(recursive_extract).collect();

        return call;
    }

    ///
    /// Returns a new expression yielding the value of the given expression. However, if the expression is a function
    /// call and should be extracted (the predicate returns true), the returned expression is just a local variable which
    /// is declared earlier and assigned the correct value. This initialization statement is added to the given vector.
    /// 
    /// Parameters:
    /// - `function_definition`: Definition of the called function
    /// - `rename_disjunct`: Closure that can rename identifiers in a way to be disjunct to all identifiers further up in the scope stack
    /// - `previous_declaration_statements`: Vector to add all initialization statements to
    /// - `prog_lifetime`: Dynamic lifetime description of the program data (as the type vector)+
    /// - `new_definitions`: Insert here all names of newly created local variables  
    ///
    pub fn extract_expression_recursive<'a, 'b, G>(
        &mut self,
        expression: Expression,
        rename_disjunct: &'a mut G,
        defined_functions: &'b DefinedFunctions,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
        prog_lifetime: Lifetime,
        new_definitions: &mut HashSet<Name>
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
                            prog_lifetime,
                            new_definitions
                        )
                    },
                    _ => Expression::Call(Box::new(self.extract_calls_in_parameters(
                        *call,
                        rename_disjunct,
                        defined_functions,
                        previous_declaration_statements,
                        prog_lifetime,
                        new_definitions
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
        prog_lifetime: Lifetime
    ) -> Vec<ExtractionReport> {

        let scopes = parent_scopes.child_scope(block);
        let mut results = Vec::new();
        let mut result_statements: Vec<Box<dyn Statement>> = Vec::new();
        let mut rename_disjunct = scopes.rename_disjunct();
        let mut new_definitions = HashSet::new();

        for mut statement in block.statements.drain(..) {
            let index_before = result_statements.len();
            for expression in statement.expressions_mut() {
                take_mut::take(expression, &mut |expr| {
                    self.extract_expression_recursive(
                        expr,
                        &mut rename_disjunct,
                        defined_functions,
                        &mut result_statements,
                        prog_lifetime,
                        &mut new_definitions
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

        let child_scope_with_new_vars = parent_scopes.child_scope(block);
        for statement in &mut block.statements {
            for subblock in statement.subblocks_mut() {
                fix_name_collisions(subblock, &child_scope_with_new_vars, &mut new_definitions, std::collections::HashMap::new());
            }
        }

        return results;
    }

    pub fn extract_calls_in_block<'a, 'b>(
        &mut self,
        block: &'a mut Block,
        parent_scopes: &'b NameScopeStack<'b>,
        defined_functions: &'b DefinedFunctions,
        prog_lifetime: Lifetime
    ) {
        let mut extractions = self.extract_calls_in_block_flat(block, parent_scopes, defined_functions, prog_lifetime);
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
            for mut subblock in statement.subblocks_mut() {
                self.extract_calls_in_block(&mut subblock, &child_scopes, defined_functions, prog_lifetime);
            }
        }
    }

    pub fn extract_calls_in_program(&mut self, (items, prog_lifetime): (&mut Vec<Box<Function>>, Lifetime)) {
        assert_ne!(items.len(), 0);
        let scopes = NameScopeStack::new(&items[..]);
        for i in 0..items.len() {
            items.swap(0, i);
            let (current, other) = items[..].split_at_mut(1);
            let child_scopes = scopes.child_scope(&*current[0]);
            if let Some(body) = &mut current[0].body {
                self.extract_calls_in_block(body, &child_scopes, other, prog_lifetime);
            }
        }
    }
}

#[cfg(test)]
use super::super::language::ast_test::*;

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

    Extractor::new(|_, _| true).extract_calls_in_program(program.work());

    let expected = Program::parse(&mut lex_str("

    fn foo(a: int,): int native;
    fn bar(a: int,): int native;

    fn main(): int {
        let result_bar: int = bar(a,);
        let result_foo: int = foo(result_bar,);
        if (result_foo) {
            let result_foo#1: int = foo(a,);
            let result_bar#1: int = bar(result_foo#1,);
            return result_bar#1;
        }
    }
    ")).unwrap();

    assert_ast_eq!(expected, program);
}

#[test]
fn test_extract_calls_new_var_name_collision_subscope() {
    let mut program = Program::parse(&mut lex_str("
    
    fn foo(a: int,): int native;

    fn main(): int {
        let x: int = foo(0,);
        {
            let result_foo: int = 1;
        }
    }
    ")).unwrap();

    Extractor::new(|_, _| true).extract_calls_in_program(program.work());

    let expected = Program::parse(&mut lex_str("

    fn foo(a: int,): int native;

    fn main(): int {
        let result_foo: int = foo(0,);
        let x: int = result_foo;
        {
            let result_foo#1: int = 1;
        }
    }
    ")).unwrap();

    assert_ast_eq!(expected, program);
}