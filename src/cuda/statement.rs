use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::super::analysis::type_error::*;
use super::expression::*;
use super::context::CudaContext;
use super::ast::*;

pub fn gen_return<'stack, 'ast: 'stack>(statement: &Return, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    Ok(Box::new(CudaReturn {
        value: statement.value.as_ref().map(|v| gen_expression(v, context)).transpose()?
    }))
}

pub fn gen_localvardef<'a, 'stack, 'ast: 'stack>(statement: &'a LocalVariableDeclaration, context: &mut dyn CudaContext<'stack, 'ast>) -> Box<dyn 'a + Iterator<Item = Result<Box<dyn CudaStatement>, OutputError>>> {
    if is_mul_var_type(&statement.declaration.variable_type) {
        let declarations = gen_variables(statement.pos(), &statement.declaration.variable, &statement.declaration.variable_type).map(|(var_type, var)|
            Ok(Box::new(CudaVarDeclaration {
                var, var_type, value: None
            }) as Box<dyn CudaStatement>)
        );
        if let Some(val) = &statement.value {
            let assignment = gen_assignment(&Assignment {
                assignee: Expression::Variable(Variable {
                    identifier: Identifier::Name(statement.declaration.variable.clone()),
                    pos: statement.pos().clone()
                }),
                pos: statement.pos().clone(),
                value: val.clone()
            }, context);
            return Box::new(declarations.chain(std::iter::once(assignment)));
        } else {
            return Box::new(declarations);
        }
    } else {
        let result = statement.value.as_ref().map(|v| gen_expression(v, context)).transpose().map(|value| {
            let (ty, var) = gen_variable(statement.pos(), &statement.declaration.variable, &statement.declaration.variable_type);
            Box::new(CudaVarDeclaration {
                var: var,
                var_type: ty,
                value: value
            }) as Box<dyn CudaStatement>
        });
        return Box::new(std::iter::once(result));
    }
}

pub fn gen_label<'stack, 'ast: 'stack>(statement: &Label, _context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    Ok(Box::new(CudaLabel {
        name: CudaIdentifier::ValueVar(statement.label.clone())
    }))
}

pub fn gen_goto<'stack, 'ast: 'stack>(statement: &Goto, _context: &mut dyn CudaContext<'stack, 'ast>) -> Result<Box<dyn CudaStatement>, OutputError> {
    Ok(Box::new(CudaGoto {
        target: CudaIdentifier::ValueVar(statement.target.clone())
    }))
}
