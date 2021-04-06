use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::{Identifier, Name};
use super::ast::*;
use super::ast_expr::*;
use super::ast_statement::*;

#[derive(Debug)]
pub struct Assignment {
    pos: TextPosition,
    pub assignee: Expression,
    pub value: Expression
}

impl PartialEq for Assignment {

    fn eq(&self, rhs: &Assignment) -> bool {
        self.value == rhs.value && self.assignee == rhs.assignee
    }
}

impl AstNodeFuncs for Assignment {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for Assignment {}

impl StatementFuncs for Assignment {
    
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(&self.value))
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.value))
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(self.assignee.names().chain(self.value.names()))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(self.assignee.names_mut().chain(self.value.names_mut()))
    }

    fn traverse_preorder<'a>(
        &'a self, 
        _parent_scopes: &DefinitionScopeStack<'_, 'a>, 
        _f: &mut dyn FnMut(&'a Block, &DefinitionScopeStack<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        Ok(())
    }

    fn traverse_preorder_mut<'a>(
        &'a mut self, 
        _parent_scopes: &DefinitionScopeStackMut<'_, 'a>, 
        _f: &mut dyn FnMut(&mut Block, &DefinitionScopeStackMut<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        Ok(())
    }
}

impl Statement for Assignment {}

impl Assignment {

    pub fn new(pos: TextPosition, assignee: Expression, value: Expression) -> Self {
        Assignment {
            pos, assignee, value
        }
    }
}