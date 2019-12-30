use super::super::parser::prelude::*;
use super::scope::*;

use std::cell::RefCell;

struct InlineTransformer 
{
    scope_stack: ScopeStack
}

impl Transformer for InlineTransformer
{
    fn before(&mut self, node: &dyn Node)
    {
        if let Some(function) = node.dynamic().downcast_ref::<FunctionNode>() {
            self.scope_stack.enter(function);
        } else if let Some(block) = node.dynamic().downcast_ref::<BlockNode>() {
            self.scope_stack.enter(block);
        }
    }

    fn after(&mut self, node: &dyn Node)
    {
        if let Some(_function) = node.dynamic().downcast_ref::<FunctionNode>() {
            self.scope_stack.exit();
        } else if let Some(_block) = node.dynamic().downcast_ref::<BlockNode>() {
            self.scope_stack.exit();
        }
    }

    fn transform_stmt(&mut self, node: Box<dyn StmtNode>) -> Box<dyn StmtNode>
    {
        unimplemented!()
    }

    fn transform_expr(&mut self, node: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
    {
        node
    }
}

pub fn inline(program: &Vec<RefCell<FunctionNode>>) 
{
    for _function in program {
        
    }
}