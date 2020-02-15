use super::super::language::prelude::*;
use super::scope::ScopeStack;

fn replace_expression_with_inlined<'a, F>(rename_disjunct: &'a mut F, result: &'a mut Vec<Box<dyn Statement>>) -> impl (FnMut(Expression) -> Expression) + 'a
    where F: FnMut(Name) -> Name
{
    move |expr| {
        match expr {
            Expression::Call(call) => {
                let pos = call.pos().clone();
                let variable_name = (*rename_disjunct)(Name::new("result".to_owned(), 0));
                let replace_params = replace_expression_with_inlined(rename_disjunct, result);
                let declaration = Declaration {
                    pos: pos.clone(),
                    value: None,
                    variable: variable_name.clone(),
                    variable_type: /* TODO: real type */ Type::Primitive(PrimitiveType::Int)
                };
                let value_block = Block {
                    pos: pos.clone(),
                    statements: vec![
                        Box::new(Assignment {
                            pos: pos.clone(),
                            assignee: Expression::Variable(Variable {
                                pos: pos.clone(),
                                identifier: Identifier::Name(variable_name.clone())
                            }),
                            value: Expression::Call(Box::new(FunctionCall {
                                pos: pos.clone(),
                                function: call.function,
                                parameters: call.parameters.into_iter().map(replace_params).collect()
                            }))
                        })
                    ]
                };
                result.push(Box::new(declaration));
                result.push(Box::new(value_block));
                return Expression::Variable(Variable {
                    pos: pos,
                    identifier: Identifier::Name(variable_name)
                });
            },
            x => x
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
#[cfg(test)]
use super::super::language::nazgul_printer::print_nazgul;

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
    let expected = Block::parse(&mut fragment_lex("{
        let result#2: int;
        {
            result#2 = (other_func)(b, );
        }
        let result#3: int;
        {
            result#3 = (c + b);
        }
        let result#1: int;
        {
            result#1 = (some_func)(result#2, result#3, );
        }
        let a: int = result#1;
    }")).unwrap();
    assert_ast_eq!(expected, block);
}