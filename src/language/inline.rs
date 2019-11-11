use super::super::parser::prelude::*;
use super::super::parser::ast_visitor::Visitable;
use super::scope::{ ScopeTable, SymbolDefinition };
use super::symbol::{ SymbolTable, SymbolDefinitionInfo, ScopeSymbolDataTransformer };
use super::super::transformer::{ Program, SpecificLifetimeTransformer, PreparedTransformer };
use super::super::util::ref_eq::ref_eq;

use std::cell::RefCell;
use std::collections::HashMap;

pub struct InlineTransformer<F>
    where F: for<'c> FnMut(&FunctionCallNode, &SymbolDefinitionInfo<'c>) -> bool
{
    should_inline: F
}

pub struct PreparedInlineTransformer<'a> {
    resolved_calls: HashMap<*const FunctionCallNode, &'a RefCell<FunctionNode>>
}

impl<'a, 'b, F> SpecificLifetimeTransformer<'a, (&'b ScopeTable<'b>, &'b SymbolTable<'b>)> for InlineTransformer<F>
    where F: for<'c> FnMut(&FunctionCallNode, &SymbolDefinitionInfo<'c>) -> bool
{
    type Prepared = PreparedInlineTransformer<'a>;

    fn prepare(mut self, program: &'a Program, (scopes, symbols): (&'b ScopeTable<'b>, &'b SymbolTable<'b>)) -> Self::Prepared {
        let mut resolved_calls: HashMap<*const FunctionCallNode, &dyn SymbolDefinition> = HashMap::new();
        for function in program {
            prepare_inline_function(&*function.borrow(), symbols, &mut self.should_inline, &mut resolved_calls);
        }
        return PreparedInlineTransformer {
            resolved_calls: unimplemented!{}
        };
    }
}

impl<'a> PreparedTransformer for PreparedInlineTransformer<'a> {
    fn transform(self, program: &Program) {

    }
}

fn prepare_inline_function<'a, F>(func: &FunctionNode, symbols: &SymbolTable<'a>, mut should_inline: F, resolved_calls: &mut HashMap<*const FunctionCallNode, &'a dyn SymbolDefinition>)
    where F: for<'b> FnMut(&FunctionCallNode, &SymbolDefinitionInfo<'b>) -> bool
{
    func.iterate(&mut |expr|{
        if let Some(function_call) = expr.dynamic().downcast_ref::<FunctionCallNode>() {
            let definition_info = symbols.get_identifier_definition(&function_call.function);
            if should_inline(function_call, definition_info) {
                let definition_node = definition_info.definition_node;
                resolved_calls.insert(function_call as *const FunctionCallNode, definition_node);
            }
        }
        return Ok(());
    });
}
