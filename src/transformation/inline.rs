use super::super::analysis::scope::NameScopeStack;
use super::super::language::prelude::*;
use super::extraction::Extractor;
use super::renaming::*;
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
        types: &mut TypeVec
    ) {
        let statement_indices_to_inline =
            self.extractor
                .extract_calls_in_block_flat(block, parent_scopes, defined_functions, types.get_lifetime());

        let scopes = parent_scopes.child_scope(block);

        {
            for extraction in &statement_indices_to_inline {
                let init_block = inline_single_function_call(block.statements[extraction.extracted_var_declaration_index]
                    .any_mut()
                    .downcast_mut::<LocalVariableDeclaration>()
                    .expect("Expected extract_calls_in_expr to generate only LocalVariableDeclaration statements"), defined_functions, &scopes, types);

                block.statements[extraction.extracted_var_value_assignment_index] =
                    Box::new(init_block);
            }
        }

        for statement in &mut block.statements {
            for subblock in statement.subblocks_mut() {
                self.inline_calls_in_block(subblock, &scopes, defined_functions, types);
            }
        }
    }

    ///
    /// Traverses the program ast and inlines all calls that match the predicate.
    ///
    pub fn inline_calls_in_program(&mut self, program: &mut Program) {
        let (items, types) = (&mut program.items, &mut program.types);
        assert_ne!(items.len(), 0);
        let scopes = NameScopeStack::new(&items[..]);
        for i in 0..items.len() {
            items.swap(0, i);
            let (current, other) = items[..].split_at_mut(1);
            let child_scopes = scopes.child_scope(&*current[0]);
            if let Some(body) = &mut current[0].body {
                self.inline_calls_in_block(body, &child_scopes, other, types);
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
    types: &mut TypeVec
) -> Block {

    let call = match std::mem::replace(&mut declaration.value, None) {
        Some(Expression::Call(call)) => call,
        _ => panic!("Expected extract_calls_in_expr to generate only LocalVariableDeclaration that are initialized with a function call result")
    };

    let definition =
        match find_function_definition(&call.function, defined_functions).internal_error() {
            FunctionDefinition::UserDefined(def) => def,
            FunctionDefinition::Builtin(builtin) => {
                panic!("Cannot inline builtin function {:?}", builtin)
            }
        };

    let pos = call.pos().clone();

    let mut rename_disjunct = scopes.rename_disjunct();

    let mut rename_mapping: HashMap<Name, Name> = HashMap::new();

    let mut result_block: Block = Block {
        pos: call.pos().clone(),
        statements: inline_parameter_passing(
            &mut rename_disjunct,
            &mut rename_mapping,
            call.parameters.into_iter(),
            definition.params.iter(),
        )
        .map(|decl| Box::new(decl) as Box<dyn Statement>)
        .collect()
    };
    let return_label_name: Name = rename_disjunct(definition.identifier.clone());

    let return_label = Label {
        pos: pos.clone(),
        label: return_label_name.clone(),
    };
    result_block.statements.push(Box::new(return_label));

    let mut body = *definition
        .body
        .as_ref()
        .expect("Cannot inline native function")
        .deep_copy_ast(types)
        .downcast_box::<Block>().unwrap();

    let in_function_body_scopes = scopes.child_scope(&result_block);

    let mut new_defs = in_function_body_scopes.non_global_definitions()
        .map(|d| d.0).chain(rename_mapping.values()).map(Name::clone).collect();

    fix_name_collisions(&mut body, &in_function_body_scopes, &mut new_defs, rename_mapping);

    replace_return_in_inlined_function_body(
        &mut body,
        &declaration.declaration.variable,
        &return_label_name
    );
    result_block.statements.insert(result_block.statements.len() - 1, Box::new(body));

    return result_block;
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

fn replace_if_type<T, F>(statement: &mut Box<dyn Statement>, f: F)
    where T: Statement, F: FnOnce(T) -> Box<dyn Statement>
{
    if statement.any().is::<T>() {
        take_mut::take(statement, |statement| {
            f(*statement.downcast_box::<T>().unwrap())
        });
    }
}

fn replace_return_in_inlined_function_body(
    block: &mut Block,
    result_variable_name: &Name,
    return_label: &Name,
) {
    for statement in &mut block.statements {
        for subblock in statement.subblocks_mut() {
            replace_return_in_inlined_function_body(
                subblock,
                result_variable_name,
                return_label
            );
        }
        replace_if_type::<Return, _>(statement, |ret| replace_return_with_assignment_and_goto(ret, result_variable_name, return_label));
    }
}

fn replace_return_with_assignment_and_goto(
    return_statement: Return,
    result_var: &Name,
    return_label: &Name,
) -> Box<dyn Statement> {

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

#[cfg(test)]
use super::super::language::ast_test::*;
#[cfg(test)]
use super::super::analysis::mock_defs::*;

#[test]

fn test_inline() {

    let parent_scope_stack: NameScopeStack = NameScopeStack::new(&[][..]);
    let predefined_variables = EnvironmentBuilder::new()
        .add_test_def("b")
        .add_test_def("c")
        .add_test_def("other_func")
        .add_test_def("some_func");

    let scope_stack = parent_scope_stack.child_scope(&predefined_variables.destruct().1);

    let mut test = Inliner::new(|_, _| true);

    let mut types = TypeVec::new();
    let mut block = Block::parse(&mut fragment_lex("
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
        &mut types
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

    assert_ast_eq!(expected, program);
}
