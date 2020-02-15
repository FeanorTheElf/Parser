use super::super::language::prelude::*;
use super::scope::ScopeStack;

fn replace_expression_with_inlined<'a, F>(rename_disjunct: &'a mut F, result: &'a mut Vec<Box<dyn Statement>>) -> impl (FnMut(Box<dyn Expression>) -> Box<dyn Expression>) + 'a
    where F: FnMut(Name) -> Name
{
    move |expr| {
        match expr.dynamic_box().downcast::<FunctionCall>() {
            Ok(call) => {
                let pos = call.pos().clone();
                let variable_name = (*rename_disjunct)(Name::new("result".to_owned(), 0));
                let replace_params = replace_expression_with_inlined(rename_disjunct, result);
                let declaration = Declaration {
                    pos: pos.clone(),
                    value: None,
                    variable: variable_name.clone(),
                    variable_type: Type::TestType
                };
                let value_block = Block {
                    pos: pos.clone(),
                    statements: vec![
                        Box::new(Assignment {
                            pos: pos.clone(),
                            assignee: Box::new(Variable {
                                pos: pos.clone(),
                                identifier: Identifier::Name(variable_name.clone())
                            }),
                            value: Box::new(FunctionCall {
                                pos: pos.clone(),
                                function: call.function,
                                parameters: call.parameters.into_iter().map(replace_params).collect()
                            })
                        })
                    ]
                };
                result.push(Box::new(declaration));
                result.push(Box::new(value_block));
                return Box::new(Variable {
                    pos: pos,
                    identifier: Identifier::Name(variable_name)
                }) as Box<dyn Expression>;
            },
            Err(expr_copy) => {
                return match expr_copy.downcast::<Variable>() {
                    Ok(var) => var as Box<dyn Expression>,
                    Err(lit) => lit.downcast::<Literal>().unwrap() as Box<dyn Expression>
                };
            }
        }
    }
}

fn prepare_inline_expressions_in_block(block: &mut Block, scopes: &ScopeStack)
{
    let mut rename_disjunct = scopes.rename_disjunct();
    let mut result_statements: Vec<Box<dyn Statement>> = Vec::new();
    for mut statement in block.statements.drain(..) {
        {
            let mut prepare_inline_expression = replace_expression_with_inlined(&mut rename_disjunct, &mut result_statements);
            for expression in statement.iter_mut() {
                take_mut::take(expression, &mut prepare_inline_expression);
            }
        }
        result_statements.push(statement);
    }
    block.statements = result_statements;
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::Parser;

#[test]
fn test_prepare_inline_expressions_in_block() {
    let mut block = Block::parse(&mut fragment_lex("{
        let a: int = some_func(other_func(b, ), c + b, );
    }")).unwrap();
    let mut scope_stack: ScopeStack = ScopeStack::new(&[]);
    let predefined_variables = [Name::l("b"), Name::l("c")];
    scope_stack.enter(&predefined_variables as &[Name]);
    scope_stack.enter(&block);
    prepare_inline_expressions_in_block(&mut block, &scope_stack);
    println!("{:?}", block);
    assert!(false);
}