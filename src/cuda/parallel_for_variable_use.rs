use super::super::analysis::scope::DefinitionScopeStack;
use super::super::analysis::symbol::SymbolDefinition;
use super::super::language::prelude::*;
use super::super::util::ref_eq::*;

use std::collections::{HashMap, HashSet};

pub struct ParallelForData<'a> {
    pub used_outer_variables: HashSet<Ref<'a, dyn SymbolDefinition>>,
}

pub fn collect_parallel_for_data_in_block<'a>(
    block: &'a Block,
    scopes: &DefinitionScopeStack<'_, 'a>,
    result: &mut HashMap<Ref<'a, ParallelFor>, ParallelForData<'a>>,
    add_uses_to: &mut Vec<ParallelForData<'a>>) 
{
    let mut child_scopes = scopes.child_scope(block);

    for statement in &block.statements {
        for expr in statement.iter() {
            collect_parallel_for_data_in_expr(expr, scopes, add_uses_to);
        }
        if let Some(parallel_for) = statement.dynamic().downcast_ref::<ParallelFor>() {
            let data = ParallelForData {
                used_outer_variables: HashSet::new(),
            };
            add_uses_to.push(data);
            for subblock in statement.iter() {
                collect_parallel_for_data_in_block(subblock, &child_scopes, result, add_uses_to);
            }
            result.insert(Ref::from(parallel_for), add_uses_to.pop().unwrap());
        } else {
            for subblock in statement.iter() {
                collect_parallel_for_data_in_block(subblock, &child_scopes, result, add_uses_to);
            }
        }
    }
}

pub fn collect_parallel_for_data_in_expr<'a>(
    expression: &'a Expression,
    scopes: &DefinitionScopeStack<'_, 'a>,
    add_uses_to: &mut Vec<ParallelForData<'a>>,
) {
    match expression {
        Expression::Call(call) => {
            for param in &call.parameters {
                collect_parallel_for_data_in_expr(param, scopes, add_uses_to);
            }
            collect_parallel_for_data_in_expr(&call.function, scopes, add_uses_to);
        }
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => {
                for parent_pfor in add_uses_to {
                    let reference: &'a dyn SymbolDefinition = *scopes.get(&name).unwrap();
                    parent_pfor
                        .used_outer_variables
                        .insert(Ref::from(reference));
                }
            }
            Identifier::BuiltIn(_) => {}
        },
        Expression::Literal(_) => {}
    }
}
