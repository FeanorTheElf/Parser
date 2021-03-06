use super::super::language::compiler::OutputError;
use super::super::language::prelude::*;
use super::ast::*;
use super::context::CudaContext;
use super::expression::*;

pub fn gen_return<'stack, 'ast: 'stack>(
    statement: &Return,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    let ast_lifetime = context.ast_lifetime();

    if is_generated_with_output_parameter(
        context.get_current_function().get_type(ast_lifetime).return_type(ast_lifetime),
    ) {

        if let Some(return_type) = context.get_current_function().get_type(ast_lifetime).return_type(ast_lifetime) {

            let assignments = std::iter::once(CudaAssignment {
                assignee: CudaExpression::deref(CudaExpression::Identifier(
                    CudaIdentifier::OutputValueVar,
                )),
                value: CudaExpression::Move(Box::new(gen_simple_expr(statement.value.as_ref().unwrap(), &*return_type).1)),
            })
            .chain(
                gen_simple_expr_array_size(statement.value.as_ref().unwrap(), &*return_type)
                    .enumerate()
                    .map(|(dim, (_ty, expr))| CudaAssignment {
                        assignee: CudaExpression::deref(CudaExpression::Identifier(
                            CudaIdentifier::OutputArraySizeVar(dim as u32),
                        )),
                        value: expr,
                    }),
            );

            Ok(Box::new(CudaBlock {
                statements: assignments
                    .map(|a| Box::new(a) as Box<dyn CudaStatement>)
                    .chain(std::iter::once(
                        Box::new(CudaReturn { value: None }) as Box<dyn CudaStatement>
                    ))
                    .collect::<Vec<_>>(),
            }))
        } else {

            Ok(Box::new(CudaReturn { value: None }))
        }
    } else {

        Ok(Box::new(CudaReturn {
            value: statement
                .value
                .as_ref()
                .map(|v| gen_expression(v, context))
                .transpose()?,
        }))
    }
}

pub fn gen_localvardef<'a, 'stack, 'ast: 'stack + 'a>(
    statement: &'a LocalVariableDeclaration,
    context: &mut dyn CudaContext<'stack, 'ast>,
) -> Box<dyn 'a + Iterator<Item = Result<Box<dyn CudaStatement>, OutputError>>> {

    let ast_lifetime = context.ast_lifetime();
    if is_mul_var_type(ast_lifetime.cast(statement.declaration.variable_type)) {

        let variable_type = ast_lifetime.cast(statement.declaration.variable_type);
        let declarations = gen_variables(
            statement.pos(),
            &statement.declaration.variable,
            &variable_type,
        )
        .map(|(var_type, var)| {

            Ok(Box::new(CudaVarDeclaration {
                var,
                var_type,
                value: None,
            }) as Box<dyn CudaStatement>)
        }).collect::<Vec<_>>().into_iter();

        if let Some(val) = &statement.value {

            let assignment = gen_assignment(
                &Assignment {
                    assignee: Expression::Variable(Variable {
                        identifier: Identifier::Name(statement.declaration.variable.clone()),
                        pos: statement.pos().clone(),
                    }),
                    pos: statement.pos().clone(),
                    value: val.clone(),
                },
                context,
            );

            return Box::new(declarations.chain(std::iter::once(assignment)));
        } else {

            return Box::new(declarations);
        }
    } else {

        let result = statement
            .value
            .as_ref()
            .map(|v| gen_expression(v, context))
            .transpose()
            .map(|value| {

                let (ty, var) = one_variable(gen_variables(
                    statement.pos(),
                    &statement.declaration.variable,
                    &*context.ast_lifetime().cast(statement.declaration.variable_type),
                ));

                Box::new(CudaVarDeclaration {
                    var: var,
                    var_type: ty,
                    value: value,
                }) as Box<dyn CudaStatement>
            });

        return Box::new(std::iter::once(result));
    }
}

pub fn gen_label<'stack, 'ast: 'stack>(
    statement: &Label,
    _context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    Ok(Box::new(CudaLabel {
        name: CudaIdentifier::ValueVar(statement.label.clone()),
    }))
}

pub fn gen_goto<'stack, 'ast: 'stack>(
    statement: &Goto,
    _context: &mut dyn CudaContext<'stack, 'ast>,
) -> Result<Box<dyn CudaStatement>, OutputError> {

    Ok(Box::new(CudaGoto {
        target: CudaIdentifier::ValueVar(statement.target.clone()),
    }))
}
