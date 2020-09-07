use super::super::analysis::scope::NameScopeStack;
use super::super::language::prelude::*;
use super::extraction::Extractor;
use super::function_resolution::{find_function_definition, DefinedFunctions, FunctionDefinition};
use std::collections::HashMap;

pub struct Inliner<F>
where
    F: FnMut(&FunctionCall, &Function) -> bool,
{
    extractor: Extractor<F>,
}

struct InlineTask<'a> {
    destination_statement_index: usize,
    result_variable_name: Name,
    call_to_inline: FunctionCall,
    function_definition: &'a Function,
}

impl<F> Inliner<F>
where
    F: FnMut(&FunctionCall, &Function) -> bool,
{
    pub fn new(should_inline: F) -> Self {

        Inliner {
            extractor: Extractor::new(should_inline),
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
        parent_scopes: &'b NameScopeStack<'b>,
        defined_functions: &'b DefinedFunctions,
    ) {

        let statement_indices_to_inline =
            self.extractor
                .extract_calls_in_block_flat(block, parent_scopes, defined_functions);

        let scopes = parent_scopes.child_scope(block);

        {

            for extraction in &statement_indices_to_inline {

                let init_block = inline_single_function_call(block.statements[extraction.extracted_var_declaration_index]
                    .dynamic_mut()
                    .downcast_mut::<LocalVariableDeclaration>()
                    .expect("Expected extract_calls_in_expr to generate only LocalVariableDeclaration statements"), defined_functions, &scopes);

                block.statements[extraction.extracted_var_value_assignment_index] =
                    Box::new(init_block);
            }
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

        let scopes = NameScopeStack::new(&program.items[..]);

        for i in 0..program.items.len() {

            program.items.swap(0, i);

            let (current, other) = program.items[..].split_at_mut(1);

            let child_scopes = scopes.child_scope(&*current[0]);

            if let Some(body) = &mut current[0].body {

                self.inline_calls_in_block(body, &child_scopes, other);
            }
        }
    }
}

///
/// Accepts a variable declaration of the form `let var: type = func(param1, ..., paramn);`
/// and inlines the call to `func`. The return value is a block containing the inlined function
/// including the assignment to `var` instead of a return statement.
///

fn inline_single_function_call(
    declaration: &mut LocalVariableDeclaration,
    defined_functions: &DefinedFunctions,
    scopes: &NameScopeStack,
) -> Block {

    let call = match std::mem::replace(&mut declaration.value, None) {
        Some(Expression::Call(call)) => call,
        _ => panic!("Expected extract_calls_in_expr to generate only LocalVariableDeclaration that are initialized with a function call result")
    };

    let definition =
        match find_function_definition(&call.function, defined_functions).internal_error() {
            FunctionDefinition::UserDefined(def) => def,
            FunctionDefinition::Builtin(builtin) => {
                panic!("Cannot inline builtin function {}", builtin)
            }
        };

    let pos = call.pos().clone();

    let mut rename_disjunct = scopes.rename_disjunct();

    let mut rename_mapping: HashMap<Name, Name> = HashMap::new();

    let mut result_statements: Vec<Box<dyn Statement>> = inline_parameter_passing(
        &mut rename_disjunct,
        &mut rename_mapping,
        call.parameters.into_iter(),
        definition.params.iter(),
    )
    .map(to_statement)
    .collect();

    let return_label_name: Name = rename_disjunct(definition.identifier.clone());

    let mut body = definition
        .body
        .as_ref()
        .expect("Cannot inline native function")
        .clone();

    process_inlined_function_body(
        &mut body,
        &declaration.declaration.variable,
        &return_label_name,
        &mut rename_disjunct,
        &mut rename_mapping,
    );

    let return_label = Label {
        pos: pos.clone(),
        label: return_label_name,
    };

    result_statements.push(Box::new(body));

    result_statements.push(Box::new(return_label));

    return Block {
        pos: pos,
        statements: result_statements,
    };
}

fn inline_parameter_passing<'a, F, I, J>(
    rename_disjunct: &'a mut F,
    rename_mapping: &'a mut HashMap<Name, Name>,
    given_params: I,
    formal_params: J,
) -> impl 'a + Iterator<Item = LocalVariableDeclaration>
where
    F: FnMut(Name) -> Name + 'a,
    I: Iterator<Item = Expression> + 'a,
    J: Iterator<Item = &'a Declaration> + 'a,
{

    given_params
        .zip(formal_params)
        .map(move |(given_param, formal_param)| {

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

            return param_decl;
        })
}

fn to_statement<T: Statement>(value: T) -> Box<dyn Statement> {

    Box::new(value)
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
use super::super::lexer::lexer::{fragment_lex, lex_str};
#[cfg(test)]
use super::super::parser::{Parser, TopLevelParser};

#[test]

fn test_inline() {

    let parent_scope_stack: NameScopeStack = NameScopeStack::new(&[]);

    let predefined_variables = [
        Name::l("b"),
        Name::l("c"),
        Name::l("other_func"),
        Name::l("some_func"),
    ];

    let scope_stack = parent_scope_stack.child_scope(&predefined_variables as &[Name]);

    let mut test = Inliner::new(|_, _| true);

    let mut types = TypeVec::new();
    let mut block = Block::parse(&mut fragment_lex(
        "
    {
        let a: int = some_func(other_func(b, ), c + b, );
        let x: int = a + 1;
    }",
    ), &mut types)
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
    ), &mut types)
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
    ), &mut types)
    .unwrap();

    let _ = test.inline_calls_in_block(
        &mut block,
        &scope_stack,
        &[Box::new(some_func), Box::new(other_func)],
    );

    let mut expected_types = TypeVec::new();
    let expected = Block::parse(&mut fragment_lex(
        "{
        let result_other_func: int;
        let result_some_func: int;
        {
            let x#1: int = b;
            {
                let i: int = 2;
                while (1) {
                    i = i + 1;
                    if ((x#1 / i) * i == x#1) {
                        {
                            result_other_func = i;
                            goto other_func#1;
                        }
                    }
                }
                {
                    result_other_func = x#1;
                    goto other_func#1;
                }
            }
            @other_func#1
        }
        {
            let a#1: int = result_other_func;
            let b#1: int = c + b;
            {
                let c#1: int = a#1;
                while (c#1 > b#1) {
                    c#1 = c#1 - b#1;
                }
                {
                    result_some_func = c#1;
                    goto some_func#1;
                }
            }
            @some_func#1
        }
        let a: int = result_some_func;
        let x: int = a + 1;
    }",
    ), &mut expected_types)
    .unwrap();

    assert_ast_frag_eq!(expected, block; expected_types.get_lifetime(), types.get_lifetime());
}

#[test]

fn test_process_inline_body() {

    let mut actual_types = TypeVec::new();
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
    ), &mut actual_types)
    .unwrap();

    let mut rename_mapping = HashMap::new();

    let parent_scopes = NameScopeStack::new(&[]);

    let scopes = parent_scopes.child_scope(&[Name::l("result")] as &[Name]);

    process_inlined_function_body(
        &mut body,
        &Name::l("result"),
        &Name::l("return_label"),
        &mut scopes.rename_disjunct(),
        &mut rename_mapping,
    );

    let mut expected_types = TypeVec::new();
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
    ), &mut expected_types)
    .unwrap();

    assert_ast_frag_eq!(expected, body; expected_types.get_lifetime(), actual_types.get_lifetime() );

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

//#[test]
fn test_inline_all() {

    let mut test = Inliner::new(|_, _| true);

    let mut program = Program::parse(&mut lex_str("
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

    let expected = Program::parse(&mut lex_str("
        fn foobar(a: int, b: int, ) {
            let result_bar: int;
            let result_foo: int;
            {
                {
                    {
                        result_bar = 4;
                        goto bar#1;
                    }
                }
                @bar#1
            }
            {
                let a#1: int = result_bar;
                {
                    let result_bar#1: int;
                    {
                        {
                            {
                                result_bar#1 = 4;
                                goto bar#1;
                            }
                        }
                        @bar#1
                    }
                    {
                        result_foo = a#1 * result_bar#1;
                        goto foo#1;
                    }
                }
                @foo#1
            }
            return result_foo;
        }

        fn foo(a: int, ): int {
            let result_bar: int;
            {
                {
                    {
                        result_bar = 4;
                        goto bar#1;
                    }
                }
                @bar#1
            }
            return a * result_bar;
        }

        fn bar(): int {
            return 4;
        }
    "))
    .unwrap();

    assert!(&expected == &program);
}
