use super::super::language::prelude::*;
use super::renaming::*;
use std::collections::{HashSet, HashMap};

pub struct Extractor<F>
where
    F: FnMut(&FunctionCall, &Function) -> bool,
{
    should_extract: F,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExtractionReport {
    pub extracted_var_value_assignment_index: usize,
    pub extracted_var_declaration_index: usize
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
    /// - `new_definitions`: Insert here all names of newly created local variables  
    ///
    fn extract_function_call<'a, 'b, G>(
        &mut self,
        function_call: FunctionCall,
        function_definition: &'b Function,
        rename_disjunct: &'a mut G,
        scopes: &DefinitionScopeStackMut,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
        new_definitions: &mut HashSet<Name>
    ) -> Result<Expression, CompileError>
    where
        G: FnMut(Name) -> Name,
        'b: 'a,
    {
        let pos = function_call.pos().clone();
        let return_type = function_definition.return_type().expect("Can only extract function calls that return a value");

        let variable_name = (*rename_disjunct)(Name::new(
            format!("result_{}", function_definition.name.name),
            0,
        ));
        new_definitions.insert(variable_name.clone());

        let declaration = LocalVariableDeclaration {
            declaration: Declaration {
                pos: pos.clone(),
                name: variable_name.clone(),
                var_type: return_type.clone(),
            },
            value: Expression::Call(Box::new(
                self.extract_calls_in_parameters(
                    function_call,
                    rename_disjunct,
                    scopes,
                    previous_declaration_statements,
                    new_definitions
                )?,
            )),
        };

        previous_declaration_statements.push(Box::new(declaration));

        return Ok(Expression::Variable(Variable {
            pos: pos,
            identifier: Identifier::Name(variable_name),
        }));
    }

    fn extract_calls_in_parameters<'a, 'b, G>(
        &mut self,
        mut call: FunctionCall,
        rename_disjunct: &'a mut G,
        defined_functions: &DefinitionScopeStackMut,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
        new_definitions: &mut HashSet<Name>
    ) -> Result<FunctionCall, CompileError>
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
                new_definitions
            )
        };

        call.parameters = call.parameters.into_iter().map(recursive_extract).collect::<Result<Vec<Expression>, CompileError>>()?;

        return Ok(call);
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
    /// - `new_definitions`: Insert here all names of newly created local variables  
    ///
    fn extract_expression_recursive<'a, 'b, G>(
        &mut self,
        expression: Expression,
        rename_disjunct: &'a mut G,
        scopes: &'b DefinitionScopeStackMut,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
        new_definitions: &mut HashSet<Name>
    ) -> Result<Expression, CompileError>
    where
        G: FnMut(Name) -> Name,
        'b: 'a,
    {
        match expression {
            Expression::Call(call) => {

                if let Identifier::Name(name) = call.function.expect_identifier()? {
                    let definition = scopes.get_defined(name, call.pos())?.downcast::<Function>().unwrap();
                    if (self.should_extract)(&call, definition) {
                        return self.extract_function_call(
                            *call,
                            definition,
                            rename_disjunct,
                            scopes,
                            previous_declaration_statements,
                            new_definitions
                        );
                    }
                }

                return Ok(Expression::Call(Box::new(self.extract_calls_in_parameters(
                    *call,
                    rename_disjunct,
                    scopes,
                    previous_declaration_statements,
                    new_definitions
                )?)));
            },
            expr => {
                return Ok(expr);
            }
        }
    }

    pub fn extract_top_level_calls_in_block<'a, 'b>(
        &mut self,
        block: &'a mut Block,
        parent_scopes: &'b DefinitionScopeStackMut
    ) -> Result<(), CompileError> {

        let mut further_defined_names = HashSet::new();
        for statement in block.statements_mut() {
            if let Some(def) = statement.as_sibling_symbol_definition_mut() {
                further_defined_names.insert(def.get_name().clone());
            }
        }

        let mut result_statements: Vec<Box<dyn Statement>> = Vec::new();
        let mut rename_disjunct = |mut name| {
            while parent_scopes.get(&name).is_some() || further_defined_names.contains(&name) {
                name.id += 1;
            }
            return name;
        };
        let mut new_definitions = HashSet::new();

        for mut statement in block.statements.drain(..) {

            for expression in statement.expressions_mut() {
                let mut error = None;
                take_mut::take(expression, &mut |expr| {
                    match self.extract_expression_recursive(
                        expr,
                        &mut rename_disjunct,
                        &parent_scopes,
                        &mut result_statements,
                        &mut new_definitions
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            error = Some(e);
                            Expression::Literal(Literal { pos: TextPosition::NONEXISTING, value: 42, literal_type: SCALAR_INT })
                        }
                    }
                });
                if let Some(e) = error {
                    return Err(e);
                }
            }

            result_statements.push(statement);
        }

        block.statements = result_statements;

        block.traverse_proper_blocks_mut(parent_scopes, &mut |block, scopes| {
            fix_name_collisions(block, scopes, &mut new_definitions, HashMap::new())
        })?;

        return Ok(());
    }

    pub fn extract_calls_in_program(&mut self, program: &mut Program) -> Result<(), CompileError> {
        program.traverse_preorder_block_mut(
            &mut |block, scopes| {
                self.extract_top_level_calls_in_block(block, scopes)?;
                return RECURSE;
            }
        )
    }
}

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

    Extractor::new(|_, _| true).extract_calls_in_program(&mut program).unwrap();

    let expected = Program::parse(&mut lex_str("

    fn foo(a: int,): int native;
    fn bar(a: int,): int native;

    fn main(): int {
        let result_bar: int init bar(a,);
        let result_foo: int init foo(result_bar,);
        if (result_foo) {
            let result_foo#1: int init foo(a,);
            let result_bar#1: int init bar(result_foo#1,);
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
        let x: int init foo(0,);
        {
            let result_foo: int init 1;
        }
    }
    ")).unwrap();

    Extractor::new(|_, _| true).extract_calls_in_program(&mut program).unwrap();

    let expected = Program::parse(&mut lex_str("

    fn foo(a: int,): int native;

    fn main(): int {
        let result_foo: int init foo(0,);
        let x: int init result_foo;
        {
            let result_foo#1: int init 1;
        }
    }
    ")).unwrap();

    assert_ast_eq!(expected, program);
}