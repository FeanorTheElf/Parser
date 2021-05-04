use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::Name;
use super::ast::*;
use super::ast_expr::*;
use super::ast_statement::*;
use super::scopes::*;
use super::symbol::*;
use super::types::*;

#[derive(Debug)]
pub struct Goto {
    pos: TextPosition,
    pub target: Name
}

impl PartialEq for Goto {

    fn eq(&self, rhs: &Goto) -> bool {
        self.target == rhs.target
    }
}

impl AstNodeFuncs for Goto {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for Goto {}

impl StatementFuncs for Goto {
    
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(std::iter::once(&self.target))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(std::iter::once(&mut self.target))
    }

    fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        f(self, parent_scopes).ignore_cancel()
    }

    fn traverse_preorder_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        f(self, parent_scopes).ignore_cancel()
    }
}

impl Statement for Goto {}

impl Goto {

    pub fn new(pos: TextPosition, target: Name) -> Self {
        Goto {
            pos, target
        }
    }
}

#[derive(Debug)]
pub struct Label {
    pos: TextPosition,
    pub name: Name
}

impl PartialEq for Label {

    fn eq(&self, rhs: &Label) -> bool {
        self.name == rhs.name
    }
}

impl AstNodeFuncs for Label {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for Label {}

impl StatementFuncs for Label {
    
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(std::iter::once(&self.name))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(std::iter::once(&mut self.name))
    }

    fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        f(self, parent_scopes).ignore_cancel()
    }

    fn traverse_preorder_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        f(self, parent_scopes).ignore_cancel()
    }
}

impl Statement for Label {}

impl Label {

    pub fn new(pos: TextPosition, name: Name) -> Self {
        Label {
            pos, name
        }
    }
}

impl SymbolDefinitionFuncs for Label {

    fn get_name(&self) -> &Name {
        &self.name
    }

    fn get_name_mut(&mut self) -> &mut Name {
        &mut self.name
    }

    fn cast_statement_mut(&mut self) -> Option<&mut dyn Statement> {
        Some(self)
    }

    fn get_type(&self) -> &Type {
        unimplemented!()
    }
}

impl SymbolDefinition for Label {}

impl SiblingSymbolDefinitionFuncs for Label {

    fn is_backward_visible(&self) -> bool {
        true
    }
}

impl SiblingSymbolDefinition for Label {}