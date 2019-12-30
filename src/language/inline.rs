use super::super::parser::prelude::*;
use super::scope::*;

use std::cell::RefCell;
use std::collections::{ HashMap, HashSet };

struct ProcessInlineBodyTransformer<'a>
{
    renaming: &'a HashMap<Identifier, Identifier>,
    result_name: &'a Identifier
}

impl<'a> Transformer for ProcessInlineBodyTransformer<'a>
{
    fn before(&mut self, _node: &dyn Node)
    {
    }

    fn after(&mut self, _node: &dyn Node)
    {
    }

    fn transform_stmt(&mut self, node: Box<dyn StmtNode>) -> Box<dyn StmtNode>
    {
        match cast::<dyn StmtNode, VariableDeclarationNode>(node) {
            Ok(mut decl) => {
                if let Some(new_name) = self.renaming.get(&decl.ident) {
                    decl.ident = new_name.clone();
                }
                decl.transform(self);
                decl
            },
            Err(node) => match cast::<dyn StmtNode, ReturnNode>(node) {
                Ok(ret) => {
                    let pos = ret.get_annotation().clone();
                    Box::new(AssignmentNode::new(pos.clone(), 
                        Box::new(ExprNodeLvlOr::new(pos.clone(),
                            Box::new(ExprNodeLvlAnd::new(pos.clone(),
                                Box::new(ExprNodeLvlCmp::new(pos.clone(),
                                    Box::new(ExprNodeLvlAdd::new(pos.clone(),
                                        Box::new(ExprNodeLvlMult::new(pos.clone(),
                                            Box::new(ExprNodeLvlIndex::new(pos.clone(),
                                                Box::new(VariableNode::new(pos, self.result_name.clone())),
                                            vec![])),
                                        vec![])),
                                    vec![])),
                                vec![])),
                            vec![])),
                        vec![])),
                    ret.expr))
                    // goto node
                },
                Err(mut node) => {
                    node.transform(self);
                    node
                }
            }
        }
    }

    fn transform_expr(&mut self, node: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
    {
        match cast::<dyn UnaryExprNode, VariableNode>(node) {
            Ok(mut var) => {
                if let Some(new_name) = self.renaming.get(&var.identifier) {
                    var.identifier = new_name.clone();
                }
                var
            },
            Err(node) => node
        }
    }
}

struct InlineTransformer 
{
    scope_stack: ScopeStack
}

impl InlineTransformer
{
    fn new(program: &Program) -> InlineTransformer
    {
        InlineTransformer {
            scope_stack: ScopeStack::new(program)
        }
    }

    fn find_variable_names<'a>(block: &'a BlockNode) -> HashSet<&'a Identifier>
    {
        let mut result = HashSet::new();
        block.iterate(&mut |node: &'a dyn Node| {
            if let Some(decl) = node.dynamic().downcast_ref::<VariableDeclarationNode>() {
                result.insert(&decl.ident);
            } else if let Some(var) = node.dynamic().downcast_ref::<VariableDeclarationNode>() {
                result.insert(&var.ident);
            }
            return Ok(());
        }).unwrap();
        return result;
    }

    fn generate_call_stmt(&self, mut expr: Box<FunctionCallNode>, function: &Box<FunctionNode>, result_name: Identifier) -> Box<dyn StmtNode>
    {
        let mut inline_body = function.implementation.dynamic().downcast_ref::<ImplementedFunctionNode>().unwrap().body.clone();
        let mut function_variable_names = Self::find_variable_names(&*inline_body);

        for param in &function.params {
            function_variable_names.insert(&param.ident);
        }

        let renaming: HashMap<Identifier, Identifier> = self.scope_stack.rename_disjunct(function_variable_names.iter().map(|name| *name))
            .map(|(original_name, new_name)| (original_name.clone(), new_name)).collect();

        inline_body.transform(&mut ProcessInlineBodyTransformer {
            renaming: &renaming,
            result_name: &result_name
        });

        let stmt_pos = expr.get_annotation().clone();
        let mut stmts: Vec<Box<dyn StmtNode>> = Vec::new();

        let parameter_count = function.params.len();
        let mut given_param_iter = expr.params.drain(..);

        for i in 0..parameter_count {
            let given_param = given_param_iter.next().unwrap();
            let formal_param = &function.params[i];
            let param_variable = Box::new(VariableDeclarationNode::new(
                given_param.get_annotation().clone(), 
                renaming.get(&formal_param.ident).unwrap().clone(), 
                formal_param.param_type.dyn_clone(), 
                Some(given_param)));
            stmts.push(param_variable);
        }
        stmts.push(inline_body);

        return Box::new(BlockNode::new(stmt_pos, stmts));
    }
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
    
}

#[cfg(test)]
use super::super::parser::Parse;
#[cfg(test)]
use super::super::lexer::lexer::lex;
#[cfg(test)]
use super::test_rename::RenameAutoVars;

#[test]
fn test_generate_call_stmt() {
    let function = FunctionNode::parse(&mut lex("fn min(a: int, b: int,): int {
            if a < b {
                return a;
            }
            return b;
        }")).unwrap();
    let expr = cast::<dyn UnaryExprNode, FunctionCallNode>(UnaryExprNode::parse(&mut lex("min(0, a + 1)")).unwrap()).unwrap();
    let expected: Box<dyn StmtNode> = StmtNode::parse(&mut lex("{
            let a: int = 0;
            let b: int = a + 1;
            {
                if a < b {
                    result = a;
                }
                result = b;
            }
        }")).unwrap();
    let mut actual: Box<dyn StmtNode> = InlineTransformer::new(&vec![]).generate_call_stmt(expr, &function, Identifier::new("result"));
    actual.rename_auto_vars();
    assert_eq!(&expected, &actual, "{} != {}", expected, actual);
}