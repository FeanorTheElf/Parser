use super::super::language::prelude::*;
use super::scope::ScopeStack;

use std::collections::HashMap;

pub struct Inliner<F>
where
    F: FnMut(&FunctionCall, &Function) -> bool,
{
    should_inline: F,
}

enum FunctionDefinition<'a> {
    Builtin(BuiltInIdentifier),
    UserDefined(&'a Function),
}

struct InlineTask<'a> {
    destination_statement_index: usize,
    result_variable_name: Name,
    call_to_inline: FunctionCall,
    function_definition: &'a Function,
}

trait ExtractFunctionCallFn<'b> =
    FnMut(Expression, &mut Vec<Box<dyn Statement>>, &mut Vec<InlineTask<'b>>) -> Expression;

type DefinedFunctions = [Box<Function>];

impl<F> Inliner<F>
where
    F: FnMut(&FunctionCall, &Function) -> bool,
{
    ///
    /// Returns a new local variable and inserts statements that initialize this variable with the result value of the
    /// given call to the given function into `previous_declaration_statements`. The other parameters are as in `inline_calls_in_expression`.
    ///
    fn extract_function_call<'a, 'b, G>(
        &'a mut self,
        function_call: FunctionCall,
        function_definition: &'b Function,
        rename_disjunct: &'a mut G,
        defined_functions: &'b DefinedFunctions,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
        open_inline_tasks: &mut Vec<InlineTask<'b>>,
    ) -> Variable
    where
        G: FnMut(Name) -> Name,
        'b: 'a,
    {
        let pos = function_call.pos().clone();
        let variable_name = (*rename_disjunct)(Name::new("result".to_owned(), 0));
        if let Some(return_type) = &function_definition.return_type {
            let declaration = LocalVariableDeclaration {
                declaration: Declaration {
                    pos: pos.clone(),
                    variable: variable_name.clone(),
                    variable_type: return_type.clone(),
                },
                value: None,
            };
            previous_declaration_statements.push(Box::new(declaration));
        }
        let inline_calls = |expression: Expression| {
            self.inline_calls_in_expression(
                expression,
                rename_disjunct,
                defined_functions,
                previous_declaration_statements,
                open_inline_tasks,
            )
        };
        let mut inline_task = InlineTask {
            destination_statement_index: 0,
            result_variable_name: variable_name.clone(),
            function_definition: function_definition,
            call_to_inline: FunctionCall {
                pos: pos.clone(),
                function: function_call.function,
                parameters: function_call
                    .parameters
                    .into_iter()
                    .map(inline_calls)
                    .collect(),
            },
        };
        let value_block = Block {
            pos: pos.clone(),
            statements: vec![],
        };
        inline_task.destination_statement_index = previous_declaration_statements.len();
        previous_declaration_statements.push(Box::new(value_block));
        open_inline_tasks.push(inline_task);
        return Variable {
            pos: pos,
            identifier: Identifier::Name(variable_name),
        };
    }

    ///
    /// Extracts all function calls in the parameters of the given call. Parameter values are as in `inline_calls_in_expression`.
    ///
    fn inline_calls_in_function_call_parameters<'a, 'b, G>(
        &'a mut self,
        mut call: FunctionCall,
        rename_disjunct: &'a mut G,
        defined_functions: &'b DefinedFunctions,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
        open_inline_tasks: &mut Vec<InlineTask<'b>>,
    ) -> FunctionCall
    where
        G: FnMut(Name) -> Name,
        'b: 'a,
    {
        let inline_calls = |expression: Expression| {
            self.inline_calls_in_expression(
                expression,
                rename_disjunct,
                defined_functions,
                previous_declaration_statements,
                open_inline_tasks,
            )
        };
        call.parameters = call.parameters.into_iter().map(inline_calls).collect();
        return call;
    }

    ///
    /// Extracts all function calls in the given expression that match the predicate. A new expression is returned,
    /// in which the extracted calls are replaced by new local variables. Any statements that have to be made to
    /// initialize these variables are added to `previous_declaration_statements`. The function execution itself
    /// is not yet inlined, as the inlining of other calls in the same scope may introduce more variables that can
    /// again cause name collisions. Instead, the corresponding statements are empty blocks, and a new task is
    /// added to `open_inline_tasks`. `rename_disjunct` is used to generate names for any new local variables.
    ///
    fn inline_calls_in_expression<'a, 'b, G>(
        &'a mut self,
        expression: Expression,
        rename_disjunct: &'a mut G,
        defined_functions: &'b DefinedFunctions,
        previous_declaration_statements: &mut Vec<Box<dyn Statement>>,
        open_inline_tasks: &mut Vec<InlineTask<'b>>,
    ) -> Expression
    where
        G: FnMut(Name) -> Name,
        'b: 'a,
    {
        match expression {
            Expression::Call(call) => {
                match find_function_definition(&call.function, defined_functions).internal_error() {
                    FunctionDefinition::UserDefined(definition)
                        if (self.should_inline)(&call, definition) =>
                    {
                        Expression::Variable(self.extract_function_call(
                            *call,
                            definition,
                            rename_disjunct,
                            defined_functions,
                            previous_declaration_statements,
                            open_inline_tasks,
                        ))
                    }
                    FunctionDefinition::UserDefined(_) => {
                        Expression::Call(Box::new(self.inline_calls_in_function_call_parameters(
                            *call,
                            rename_disjunct,
                            defined_functions,
                            previous_declaration_statements,
                            open_inline_tasks,
                        )))
                    }
                    _ => Expression::Call(Box::new(self.inline_calls_in_function_call_parameters(
                        *call,
                        rename_disjunct,
                        defined_functions,
                        previous_declaration_statements,
                        open_inline_tasks,
                    ))),
                }
            }
            expr => expr,
        }
    }

    ///
    /// Inlines all function calls in expressions that occur in the given block and that match the predicate.
    /// The `parent_scopes` parameter must be a scope stack describing the whole context up to this block, not
    /// including it which is used to resolve called functions and prevent name collisions. The given
    /// `defined_functions` must include any function definition that might be called from within this block
    /// (excluding builtin functions), and is used to retrieve the body to inline.
    ///
    fn inline_calls_in_block<'a, 'b>(
        &mut self,
        block: &'a mut Block,
        parent_scopes: &'b ScopeStack<'b>,
        defined_functions: &'b DefinedFunctions,
    ) {
        let mut scopes = parent_scopes.child_stack();
        let mut inline_tasks: Vec<InlineTask> = vec![];
        scopes.enter(block);
        {
            let mut rename_disjunct = scopes.rename_disjunct();
            let mut result_statements: Vec<Box<dyn Statement>> = Vec::new();
            for mut statement in block.statements.drain(..) {
                for expression in statement.iter_mut() {
                    take_mut::take(expression, &mut |expr| {
                        self.inline_calls_in_expression(
                            expr,
                            &mut rename_disjunct,
                            defined_functions,
                            &mut result_statements,
                            &mut inline_tasks,
                        )
                    });
                }
                result_statements.push(statement);
            }
            block.statements = result_statements;
        }

        scopes.exit();
        scopes.enter(block);

        for task in inline_tasks.into_iter() {
            let block_index = task.destination_statement_index;
            inline_single_function_call(
                task.call_to_inline,
                task.function_definition,
                task.result_variable_name,
                block.statements[block_index]
                    .dynamic_mut()
                    .downcast_mut::<Block>()
                    .unwrap(),
                &scopes,
            );
        }

        for statement in &mut block.statements {
            for subblock in statement.iter_mut() {
                self.inline_calls_in_block(subblock, &scopes, defined_functions);
            }
        }
    }

    ///
    /// Traverses the program ast and inlines all calls that match the predicate.
    ///
    pub fn inline_calls_in_program(&mut self, program: &mut Program) {
        assert_ne!(program.items.len(), 0);
        let mut scopes = ScopeStack::new(&program.items[..]);
        for i in 0..program.items.len() {
            program.items.swap(0, i);
            let (current, other) = program.items[..].split_at_mut(1);
            scopes.enter(&*current[0]);
            if let Some(body) = &mut current[0].body {
                self.inline_calls_in_block(body, &scopes, other);
            }
            scopes.exit();
        }
    }
}

fn find_function_definition<'b>(
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

fn inline_single_function_call(
    call: FunctionCall,
    definition: &Function,
    result_var: Name,
    block: &mut Block,
    scopes: &ScopeStack,
) {
    let pos = call.pos;
    let mut rename_disjunct = scopes.rename_disjunct();
    let mut rename_mapping: HashMap<Name, Name> = HashMap::new();
    for (given_param, formal_param) in call.parameters.into_iter().zip(definition.params.iter()) {
        let param_name = rename_disjunct(formal_param.variable.clone());
        rename_mapping.insert(formal_param.variable.clone(), param_name.clone());
        let param_type = formal_param.variable_type.clone();
        let param_value = given_param;
        let param_decl = LocalVariableDeclaration {
            declaration: Declaration {
                pos: param_value.pos().clone(),
                variable: param_name,
                variable_type: param_type,
            },
            value: Some(param_value),
        };
        block.statements.push(Box::new(param_decl))
    }
    let finish_label: Name = rename_disjunct(definition.identifier.clone());
    let mut body = definition
        .body
        .as_ref()
        .expect("Cannot inline native function")
        .clone();
    process_inlined_function_body(
        &mut body,
        &result_var,
        &finish_label,
        &mut rename_disjunct,
        &mut rename_mapping,
    );
    block.statements.push(Box::new(body));
    block.statements.push(Box::new(Label {
        pos: pos,
        label: finish_label,
    }));
}

fn process_inlined_function_body<F>(
    block: &mut Block,
    result_variable_name: &Name,
    return_label: &Name,
    rename_disjunct: &mut F,
    rename_mapping: &mut HashMap<Name, Name>,
) where
    F: FnMut(Name) -> Name,
{
    for statement in &mut block.statements {
        for subblock in statement.iter_mut() {
            process_inlined_function_body(
                subblock,
                result_variable_name,
                return_label,
                rename_disjunct,
                rename_mapping,
            );
        }
        for expression in statement.iter_mut() {
            recursive_rename_variables(expression, rename_disjunct, rename_mapping);
        }
        if statement.dynamic().is::<Return>() {
            take_mut::take(statement, |statement| {
                transform_inlined_return_statement(statement, result_variable_name, return_label)
            });
        } else if let Some(declaration) = statement
            .dynamic_mut()
            .downcast_mut::<LocalVariableDeclaration>()
        {
            declaration.declaration.variable = rename_identifier(
                &declaration.declaration.variable,
                rename_disjunct,
                rename_mapping,
            );
        } else if let Some(goto) = statement.dynamic_mut().downcast_mut::<Goto>() {
            goto.target = rename_identifier(&goto.target, rename_disjunct, rename_mapping);
        } else if let Some(label) = statement.dynamic_mut().downcast_mut::<Label>() {
            label.label = rename_identifier(&label.label, rename_disjunct, rename_mapping);
        }
    }
}

fn transform_inlined_return_statement(
    statement: Box<dyn Statement>,
    result_var: &Name,
    return_label: &Name,
) -> Box<dyn Statement> {
    let return_statement = statement.dynamic_box().downcast::<Return>().unwrap();
    let pos = return_statement.pos().clone();
    let mut result = Box::new(Block {
        pos: pos.clone(),
        statements: vec![],
    });
    if let Some(return_value) = return_statement.value {
        let assign_return_value = Assignment {
            pos: pos.clone(),
            assignee: Expression::Variable(Variable {
                pos: pos.clone(),
                identifier: Identifier::Name(result_var.clone()),
            }),
            value: return_value,
        };
        result.statements.push(Box::new(assign_return_value));
    }
    let jump_out = Goto {
        pos: pos,
        target: return_label.clone(),
    };
    result.statements.push(Box::new(jump_out));
    return result;
}

fn recursive_rename_variables<F>(
    expr: &mut Expression,
    rename_disjunct: &mut F,
    rename_mapping: &mut HashMap<Name, Name>,
) where
    F: FnMut(Name) -> Name,
{
    match expr {
        Expression::Call(call) => {
            recursive_rename_variables(&mut call.function, rename_disjunct, rename_mapping);
            for param in &mut call.parameters {
                recursive_rename_variables(param, rename_disjunct, rename_mapping);
            }
        }
        Expression::Literal(_) => {}
        Expression::Variable(var) => {
            if let Identifier::Name(name) = &mut var.identifier {
                *name = rename_identifier(&name, rename_disjunct, rename_mapping);
            }
        }
    }
}

fn rename_identifier<F>(
    name: &Name,
    rename_disjunct: &mut F,
    rename_mapping: &mut HashMap<Name, Name>,
) -> Name
where
    F: FnMut(Name) -> Name,
{
    if let Some(replace) = rename_mapping.get(&name) {
        return replace.clone();
    } else {
        let replace = rename_disjunct(name.clone());
        rename_mapping.insert(name.clone(), replace.clone());
        return replace;
    }
}

#[cfg(test)]
use super::super::language::debug_printer::print_debug;
#[cfg(test)]
use super::super::lexer::lexer::{fragment_lex, lex};
#[cfg(test)]
use super::super::parser::Parser;

#[test]
fn test_inline() {
    let mut scope_stack: ScopeStack = ScopeStack::new(&[]);
    let predefined_variables = [
        Name::l("b"),
        Name::l("c"),
        Name::l("other_func"),
        Name::l("some_func"),
    ];
    scope_stack.enter(&predefined_variables as &[Name]);
    let mut test = Inliner {
        should_inline: |_, _| true,
    };

    let mut block = Block::parse(&mut fragment_lex(
        "
    {
        let a: int = some_func(other_func(b, ), c + b, );
        let x: int = a + 1;
    }",
    ))
    .unwrap();

    let some_func: Function = Function::parse(&mut fragment_lex(
        "
    fn some_func(a: int, b: int, ): int  {
        let c: int = a;
        while (c > b) {
            c = c - b;
        }
        return c;
    }",
    ))
    .unwrap();

    let other_func: Function = Function::parse(&mut fragment_lex(
        "
    fn other_func(x: int, ): int {
        let i: int = 2;
        while (1) {
            i = i + 1;
            if ((x / i) * i == x) {
                return i;
            }
        }
        return x;
    }",
    ))
    .unwrap();

    let _ = test.inline_calls_in_block(
        &mut block,
        &scope_stack,
        &[Box::new(some_func), Box::new(other_func)],
    );

    let expected = Block::parse(&mut fragment_lex(
        "{
        let result#0: int;
        let result#1: int;
        {
            let x#1: int = b;
            {
                let i: int = 2;
                while (1) {
                    i = i + 1;
                    if ((x#1 / i) * i == x#1) {
                        {
                            result#1 = i;
                            goto other_func#1;
                        }
                    }
                }
                {
                    result#1 = x#1;
                    goto other_func#1;
                }
            }
            @other_func#1
        }
        {
            let a#1: int = result#1;
            let b#1: int = c + b;
            {
                let c#1: int = a#1;
                while (c#1 > b#1) {
                    c#1 = c#1 - b#1;
                }
                {
                    result#0 = c#1;
                    goto some_func#1;
                }
            }
            @some_func#1
        }
        let a: int = result#0;
        let x: int = a + 1;
    }",
    ))
    .unwrap();
    assert_ast_eq!(expected, block);
}

#[test]
fn test_process_inline_body() {
    let mut body = Block::parse(&mut fragment_lex(
        "
    {
        let result: int = 10;
        while (1) {
            if (result < 5) {
                if (result == 0) {
                    return result;
                    goto result#1;
                }
                {
                    result = result - 1;
                    return result;
                }
                @result#1
            }
        }
    }",
    ))
    .unwrap();
    let mut rename_mapping = HashMap::new();
    let mut scopes = ScopeStack::new(&[]);
    scopes.enter(&[Name::l("result")] as &[Name]);
    process_inlined_function_body(
        &mut body,
        &Name::l("result"),
        &Name::l("return_label"),
        &mut scopes.rename_disjunct(),
        &mut rename_mapping,
    );

    let expected = Block::parse(&mut fragment_lex(
        "
    {
        let result#1: int = 10;
        while (1) {
            if (result#1 < 5) {
                if (result#1 == 0) {
                    {
                        result = result#1;
                        goto return_label;
                    }
                    goto result#2;
                }
                {
                    result#1 = result#1 - 1;
                    {
                        result = result#1;
                        goto return_label;
                    }
                }
                @result#2
            }
        }
    }",
    ))
    .unwrap();
    assert_ast_eq!(expected, body);

    let mut expected_mapping = HashMap::new();
    expected_mapping.insert(
        Name::new("result".to_owned(), 0),
        Name::new("result".to_owned(), 1),
    );
    expected_mapping.insert(
        Name::new("result".to_owned(), 1),
        Name::new("result".to_owned(), 2),
    );
    assert_eq!(expected_mapping, rename_mapping);
}

#[test]
fn test_inline_all() {
    let mut test = Inliner {
        should_inline: |_, _| true,
    };

    let mut program = Program::parse(&mut lex("
        fn foo(a: int, ): int {
            return a * bar();
        }

        fn bar(): int {
            return 4;
        }

        fn foobar(a: int, b: int, ) {
            return foo(bar(),);
        }
    "))
    .unwrap();
    test.inline_calls_in_program(&mut program);

    let expected = Program::parse(&mut lex("
        fn foobar(a: int, b: int, ) {
            let result: int;
            let result#1: int;
            {
                {
                    {
                        result#1 = 4;
                        goto bar#1;
                    }
                }
                @bar#1
            }
            {
                let a#1: int = result#1;
                {
                    let result#2: int;
                    {
                        {
                            {
                                result#2 = 4;
                                goto bar#1;
                            }
                        }
                        @bar#1
                    }
                    {
                        result = a#1 * result#2;
                        goto foo#1;
                    }
                }
                @foo#1
            }
            return result;
        }

        fn foo(a: int, ): int {
            let result: int;
            {
                {
                    {
                        result = 4;
                        goto bar#1;
                    }
                }
                @bar#1
            }
            return a * result;
        }

        fn bar(): int {
            return 4;
        }
    "))
    .unwrap();
    assert_ast_eq!(expected, program);
}
