use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::Name;
use super::ast::*;
use super::ast_expr::*;
use super::ast_statement::*;

#[derive(Debug)]
pub struct Return {
    pos: TextPosition,
    value: Expression
}

impl PartialEq for Return {

    fn eq(&self, rhs: &Return) -> bool {
        self.value == rhs.value
    }
}

impl AstNodeFuncs for Return {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for Return {}

impl StatementFuncs for Return {
    
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
        Box::new(self.value.names())
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(self.value.names_mut())
    }

    fn traverse_preorder<'a>(
        &'a self, 
        _parent_scopes: &DefinitionScopeStack, 
        _f: &mut dyn FnMut(&'a Block, &DefinitionScopeStack) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        Ok(())
    }

    fn traverse_preorder_mut(
        &mut self, 
        _parent_scopes: &DefinitionScopeStackMut, 
        _f: &mut dyn FnMut(&mut Block, &DefinitionScopeStackMut) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        Ok(())
    }
}

