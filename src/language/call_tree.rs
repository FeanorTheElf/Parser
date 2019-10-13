use super::super::parser::prelude::*;
use super::super::parser::ast_visitor::Visitable;
use super::symbol::{ SymbolTable, SymbolUse };
use super::super::util::ref_eq::{ ref_eq, Ref };

use std::cell::RefCell;
use std::collections::HashMap;

pub struct CallTreeNode<'a> {
    pub function: &'a RefCell<FunctionNode>,
    pub children: Vec<Option<Box<CallTreeNode<'a>>>>
}

pub fn calc_call_tree<'a>(functions: &'a Vec<&'a FunctionNode>, symbols: &SymbolTable<'a>) -> Result<CallTreeNode<'a>, CompileError> {
    let call_graph = calc_call_graph(functions, symbols);    
    unimplemented!();
}

fn calc_call_graph<'a>(functions: &'a Vec<&'a FunctionNode>, symbols: &SymbolTable<'a>) -> HashMap<Ref<'a, &'a FunctionNode>, Vec<&'a FunctionNode>> {
    let find_corresponding_vector_entry = |node: &'a FunctionNode| *functions.iter().find(|f| ref_eq(**f, node)).unwrap();

    let mut result = HashMap::new();
    for function in functions.iter() {
        if let Some(implementation) = function.implementation.dynamic().downcast_ref::<ImplementedFunctionNode>() {
            let called_functions: Vec<&'a FunctionNode> = find_called_functions(implementation, symbols).iter()
                .map(|call|*call).map(find_corresponding_vector_entry).collect();
            result.insert(Ref::from(function), called_functions);
        }
    }
    return result;
}

fn find_called_functions<'a>(implementation: &'a ImplementedFunctionNode, symbols: &SymbolTable<'a>) -> Vec<&'a FunctionNode> {
    let mut result = Vec::new();
    implementation.stmts.iterate(&mut |unary_expr: &'a dyn UnaryExprNode| {
        if let Some(function_call) = unary_expr.dynamic().downcast_ref::<FunctionCallNode>() {
            let called_function: &'a FunctionNode = symbols.get_identifier_definition(function_call.get_identifier()).definition_node
                .dynamic().downcast_ref::<FunctionNode>().unwrap();
            result.push(called_function);
        }
        return Ok(());
    }).unwrap();
    return result;
}