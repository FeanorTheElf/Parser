use super::super::parser::prelude::*;
use super::super::util::ref_eq::ref_eq;

use std::cell::RefCell;
use std::collections::HashMap;

pub fn inline(program: &Vec<RefCell<FunctionNode>>) 
{
    for function in program {
        function.borrow_mut().transform(&mut |stmt: Box<dyn StmtNode>| {
            match cast::<dyn StmtNode, VariableDeclarationNode>(stmt) {
                Ok(call) => {
                    unimplemented!()
                },
                Err(stmt) => stmt
            }
        });
    }
}