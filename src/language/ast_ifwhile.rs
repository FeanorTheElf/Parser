use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::Name;
use super::ast::*;
use super::ast_expr::*;
use super::ast_statement::*;
use super::scopes::*;

#[derive(Debug)]
pub struct If {
    pos: TextPosition,
    pub condition: Expression,
    pub body: Block
}

impl PartialEq for If {

    fn eq(&self, rhs: &If) -> bool {
        self.condition == rhs.condition && self.body == rhs.body
    }
}

impl AstNodeFuncs for If {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for If {}

impl StatementFuncs for If {
    
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(&self.body))
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(&mut self.body))
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(&self.condition))
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.condition))
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(self.condition.names().chain(self.body.names()))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(self.condition.names_mut().chain(self.body.names_mut()))
    }

    fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        let recurse = f(self, parent_scopes);
        self.body.traverse_preorder_base(parent_scopes, f, recurse)
    }

    fn traverse_preorder_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        let recurse = f(self, parent_scopes);
        self.body.traverse_preorder_mut_base(parent_scopes, f, recurse)
    }
}

impl Statement for If {}

impl If {

    pub fn new(pos: TextPosition, condition: Expression, body: Block) -> Self {
        If {
            pos, condition, body
        }
    }
}

#[derive(Debug)]
pub struct While {
    pos: TextPosition,
    pub condition: Expression,
    pub body: Block
}

impl PartialEq for While {

    fn eq(&self, rhs: &While) -> bool {
        self.condition == rhs.condition && self.body == rhs.body
    }
}

impl AstNodeFuncs for While {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for While {}


impl StatementFuncs for While {
    
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(&self.body))
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(&mut self.body))
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(&self.condition))
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.condition))
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(self.condition.names().chain(self.body.names()))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(self.condition.names_mut().chain(self.body.names_mut()))
    }

    fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        let recurse = f(self, parent_scopes);
        self.body.traverse_preorder_base(parent_scopes, f, recurse)
    }

    fn traverse_preorder_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        let recurse = f(self, parent_scopes);
        self.body.traverse_preorder_mut_base(parent_scopes, f, recurse)
    }
}

impl Statement for While {}

impl While {

    pub fn new(pos: TextPosition, condition: Expression, body: Block) -> Self {
        While {
            pos, condition, body
        }
    }
}