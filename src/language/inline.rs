use super::super::parser::prelude::*;
use super::super::parser::ast_visitor::Visitable;
use super::scope::{ ScopeTable, SymbolDefinition };
use super::symbol::{ SymbolTable, SymbolDefinitionInfo };
use super::super::util::ref_eq::ref_eq;

use std::cell::RefCell;
use std::collections::HashMap;

pub struct InlinePreparation<'b> {
    resolved_calls: HashMap<*const FunctionCallNode, &'b RefCell<FunctionNode>>
}

pub fn prepare_inline<'a, 'c, F>(funcs: &'c Vec<RefCell<FunctionNode>>, symbols: &SymbolTable<'a>, mut should_inline: F) -> InlinePreparation<'c>
    where F: for<'b> FnMut(&FunctionCallNode, &SymbolDefinitionInfo<'b>) -> bool
{
    let mut resolved_calls: HashMap<*const FunctionCallNode, &dyn SymbolDefinition> = HashMap::new();
    for function in funcs {
        prepare_inline_function(&*function.borrow(), symbols, &mut should_inline, &mut resolved_calls);
    }
    return InlinePreparation {
        resolved_calls: unimplemented!{}
    };
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

pub fn perform_inline<'b>(funcs: &Vec<RefCell<FunctionCallNode>>, data: InlinePreparation<'b>) {

}