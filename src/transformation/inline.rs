use super::super::language::prelude::*;
use super::scope::ScopeStack;

use std::collections::HashMap;

pub struct Inliner<F>
    where F: FnMut(&FunctionCall, &Function) -> bool
{
    should_inline: F
}

enum FunctionDefinition<'a>
{
    Builtin(BuiltInIdentifier),
    UserDefined(&'a Function)
}

struct InlineTask<'a>
{
    destination_statement_index: usize,
    result_variable_name: Name,
    call_to_inline: FunctionCall,
    function_definition: &'a Function
}

trait ExtractFunctionCallFn<'b> = FnMut(Expression, &mut Vec<Box<dyn Statement>>, &mut Vec<InlineTask<'b>>) -> Expression;

type DefinedFunctions = [Box<Function>];

impl<F> Inliner<F>
    where F: FnMut(&FunctionCall, &Function) -> bool
{
    fn extract_function_call<'a, 'b, G>(&'a mut self, 
            rename_disjunct: &'a mut G, 
            defined_functions: &'b DefinedFunctions, 
            function_call_to_extract: Expression, 
            previous_declaration_statements: &mut Vec<Box<dyn Statement>>, 
            open_inline_tasks: &mut Vec<InlineTask<'b>>) -> Expression
        where G: FnMut(Name) -> Name, 'b: 'a
    {
            match function_call_to_extract {
                Expression::Call(mut call) => {
                    let called_function = find_function_definition(&call.function, defined_functions).internal_error();
                    let pos = call.pos().clone();
                    if let FunctionDefinition::UserDefined(function_def) = called_function {
                        if (self.should_inline)(&call, function_def) {
                            let variable_name = (*rename_disjunct)(Name::new("result".to_owned(), 0));
                            if let Some(return_type) = &function_def.return_type {
                                let declaration = LocalVariableDeclaration {
                                    declaration: Declaration {
                                        pos: pos.clone(),
                                        variable: variable_name.clone(),
                                        variable_type: return_type.clone()
                                    },
                                    value: None
                                };
                                previous_declaration_statements.push(Box::new(declaration));
                            }
                            let mut inline_task = InlineTask {
                                destination_statement_index: 0,
                                result_variable_name: variable_name.clone(),
                                function_definition: function_def,
                                call_to_inline: FunctionCall {
                                    pos: pos.clone(),
                                    function: call.function,
                                    parameters: call.parameters.into_iter().map(|p| 
                                        self.extract_function_call(rename_disjunct, defined_functions, p, previous_declaration_statements, open_inline_tasks)).collect()
                                }
                            };
                            let value_block = Block {
                                pos: pos.clone(),
                                statements: vec![]
                            };
                            inline_task.destination_statement_index = previous_declaration_statements.len();
                            previous_declaration_statements.push(Box::new(value_block));
                            open_inline_tasks.push(inline_task);
                            return Expression::Variable(Variable {
                                pos: pos,
                                identifier: Identifier::Name(variable_name)
                            });
                        } else {
                            call.parameters = call.parameters.into_iter().map(|p| 
                                self.extract_function_call(rename_disjunct, defined_functions, p, previous_declaration_statements, open_inline_tasks)).collect();
                            return Expression::Call(call);
                        }
                    } else {
                        call.parameters = call.parameters.into_iter().map(|p| 
                            self.extract_function_call(rename_disjunct, defined_functions, p, previous_declaration_statements, open_inline_tasks)).collect();
                        return Expression::Call(call);
                    }
                },
                expr => {
                    return expr;
                }
            }
    }

    fn inline<'a, 'b>(&mut self, block: &'a mut Block, parent_scopes: &'b ScopeStack<'b>, defined_functions: &'b DefinedFunctions)
    {
        let mut scopes = parent_scopes.child_stack();
        let mut inline_tasks: Vec<InlineTask> = vec![];
        scopes.enter(block);
        {
            let mut rename_disjunct = scopes.rename_disjunct();
            let mut result_statements: Vec<Box<dyn Statement>> = Vec::new();
            for mut statement in block.statements.drain(..) {
                for expression in statement.iter_mut() {
                    take_mut::take(expression, &mut |expr| 
                        self.extract_function_call(&mut rename_disjunct, defined_functions, expr, &mut result_statements, &mut inline_tasks));
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
                block.statements[block_index].dynamic_mut().downcast_mut::<Block>().unwrap(), &scopes);
        }
        for statement in &mut block.statements {
            for subblock in statement.iter_mut() {
                self.inline(subblock, &scopes, defined_functions);
            }
        }
    }

    pub fn inline_all(&mut self, program: &mut Program)
    {
        assert_ne!(program.items.len(), 0);
        let mut scopes = ScopeStack::new(&program.items[..]);
        for i in 0..program.items.len() {
            program.items.swap(0, i);
            let (current, other) = program.items[..].split_at_mut(1);
            scopes.enter(&*current[0]);
            if let Some(body) = &mut current[0].body {
                self.inline(body, &scopes, other);
            }
            scopes.exit();
        }
    }
}

fn find_function_definition<'b>(expr: &Expression, all_functions: &'b DefinedFunctions) -> Result<FunctionDefinition<'b>, CompileError>
{
    if let Expression::Variable(identifier) = expr {
        match &identifier.identifier {
            Identifier::Name(name) => {
                let found_function = all_functions.iter().find(|f| f.identifier == *name).ok_or(
                    CompileError::new(expr.pos(), format!("Could not find definition of function {}", name), ErrorType::UndefinedSymbol));
                Ok(FunctionDefinition::UserDefined(found_function?))
            },
            Identifier::BuiltIn(builtin_identifier) => Ok(FunctionDefinition::Builtin(*builtin_identifier))
        }
    } else {
        panic!("Cannot inline calls to functions that are the result of an expression");
    }
}

fn inline_single_function_call(call: FunctionCall, definition: &Function, result_var: Name, block: &mut Block, scopes: &ScopeStack)
{
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
                variable_type: param_type
            },
            value: Some(param_value)
        };
        block.statements.push(Box::new(param_decl))
    }
    let finish_label: Name = rename_disjunct(definition.identifier.clone());
    let mut body = definition.body.as_ref().expect("Cannot inline native function").clone();
    process_inlined_function_body(&mut body, &result_var, &finish_label, &mut rename_disjunct, &mut rename_mapping);
    block.statements.push(Box::new(body));
    block.statements.push(Box::new(Label {
        pos: pos,
        label: finish_label
    }));
}

fn process_inlined_function_body<F>(block: &mut Block, result_variable_name: &Name, return_label: &Name, rename_disjunct: &mut F, rename_mapping: &mut HashMap<Name, Name>)
    where F: FnMut(Name) -> Name
{
    for statement in &mut block.statements {
        for subblock in statement.iter_mut() {
            process_inlined_function_body(subblock, result_variable_name, return_label, rename_disjunct, rename_mapping);
        }
        for expression in statement.iter_mut() {
            recursive_rename_variables(expression, rename_disjunct, rename_mapping);
        }
        if statement.dynamic().is::<Return>() {
            take_mut::take(statement, |statement| transform_inlined_return_statement(statement, result_variable_name, return_label));
        } else if let Some(declaration) = statement.dynamic_mut().downcast_mut::<LocalVariableDeclaration>() {
            declaration.declaration.variable = rename_identifier(&declaration.declaration.variable, rename_disjunct, rename_mapping);
        } else if let Some(goto) = statement.dynamic_mut().downcast_mut::<Goto>() {
            goto.target = rename_identifier(&goto.target, rename_disjunct, rename_mapping);
        } else if let Some(label) = statement.dynamic_mut().downcast_mut::<Label>() {
            label.label = rename_identifier(&label.label, rename_disjunct, rename_mapping);
        }
    }
}

fn transform_inlined_return_statement(statement: Box<dyn Statement>, result_var: &Name, return_label: &Name) -> Box<dyn Statement>
{
    let return_statement = statement.dynamic_box().downcast::<Return>().unwrap();
    let pos = return_statement.pos().clone();
    let mut result = Box::new(Block {
        pos: pos.clone(),
        statements: vec![]
    });
    if let Some(return_value) = return_statement.value {
        let assign_return_value = Assignment {
            pos: pos.clone(),
            assignee: Expression::Variable(Variable {
                pos: pos.clone(),
                identifier: Identifier::Name(result_var.clone())
            }),
            value: return_value
        };
        result.statements.push(Box::new(assign_return_value));
    }
    let jump_out = Goto {
        pos: pos,
        target: return_label.clone()
    };
    result.statements.push(Box::new(jump_out));
    return result;
}

fn recursive_rename_variables<F>(expr: &mut Expression, rename_disjunct: &mut F, rename_mapping: &mut HashMap<Name, Name>)
    where F: FnMut(Name) -> Name
{
    match expr {
        Expression::Call(call) => {
            recursive_rename_variables(&mut call.function, rename_disjunct, rename_mapping);
            for param in &mut call.parameters {
                recursive_rename_variables(param, rename_disjunct, rename_mapping);
            }
        },
        Expression::Literal(_) => { },
        Expression::Variable(var) => {
            if let Identifier::Name(name) = &mut var.identifier {
                *name = rename_identifier(&name, rename_disjunct, rename_mapping);
            }
        }
    }
}

fn rename_identifier<F>(name: &Name, rename_disjunct: &mut F, rename_mapping: &mut HashMap<Name, Name>) -> Name
    where F: FnMut(Name) -> Name
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
use super::super::lexer::lexer::{ fragment_lex, lex };
#[cfg(test)]
use super::super::parser::Parser;
#[cfg(test)]
use super::super::language::debug_printer::print_debug;

#[test]
fn test_inline() {
    let mut scope_stack: ScopeStack = ScopeStack::new(&[]);
    let predefined_variables = [Name::l("b"), Name::l("c"), Name::l("other_func"), Name::l("some_func")];
    scope_stack.enter(&predefined_variables as &[Name]);
    let mut test = Inliner { should_inline: |_, _| true };

    let mut block = Block::parse(&mut fragment_lex("
    {
        let a: int = some_func(other_func(b, ), c + b, );
        let x: int = a + 1;
    }")).unwrap();

    let some_func: Function = Function::parse(&mut fragment_lex("
    fn some_func(a: int, b: int, ): int  {
        let c: int = a;
        while (c > b) {
            c = c - b;
        }
        return c;
    }")).unwrap();

    let other_func: Function = Function::parse(&mut fragment_lex("
    fn other_func(x: int, ): int {
        let i: int = 2;
        while (1) {
            i = i + 1;
            if ((x / i) * i == x) {
                return i;
            }
        }
        return x;
    }")).unwrap();

    let _ = test.inline(&mut block, &scope_stack, &[Box::new(some_func), Box::new(other_func)]);

    let expected = Block::parse(&mut fragment_lex("{
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
    }")).unwrap();
    assert_ast_eq!(expected, block);
}

#[test]
fn test_process_inline_body() {
    let mut body = Block::parse(&mut fragment_lex("
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
    }")).unwrap();
    let mut rename_mapping = HashMap::new();
    let mut scopes = ScopeStack::new(&[]);
    scopes.enter(&[Name::l("result")] as &[Name]);
    process_inlined_function_body(&mut body, &Name::l("result"), &Name::l("return_label"), &mut scopes.rename_disjunct(), &mut rename_mapping);

    let expected = Block::parse(&mut fragment_lex("
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
    }")).unwrap();
    assert_ast_eq!(expected, body);

    let mut expected_mapping = HashMap::new();
    expected_mapping.insert(Name::new("result".to_owned(), 0), Name::new("result".to_owned(), 1));
    expected_mapping.insert(Name::new("result".to_owned(), 1), Name::new("result".to_owned(), 2));
    assert_eq!(expected_mapping, rename_mapping);
}

#[test]
fn test_inline_all() {
    let mut test = Inliner { should_inline: |_, _| true };

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
    ")).unwrap();
    test.inline_all(&mut program);

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
    ")).unwrap();
    assert_ast_eq!(expected, program);
}