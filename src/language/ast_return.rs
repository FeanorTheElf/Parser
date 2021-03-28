use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::Name;
use super::ast::*;
use super::ast_expr::*;
use super::ast_statement::*;

#[derive(Debug)]
pub struct Return {
    pos: TextPosition,
    pub value: Option<Expression>
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
        Box::new(std::iter::once(&self.value).filter_map(|x| x.as_ref()))
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.value).filter_map(|x| x.as_mut()))
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(std::iter::once(&self.value).filter_map(|x| x.as_ref()).flat_map(|x|  x.names()))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(std::iter::once(&mut self.value).filter_map(|x| x.as_mut()).flat_map(|x|  x.names_mut()))
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

impl Statement for Return {}

impl Return {

    pub fn new(pos: TextPosition, value: Option<Expression>) -> Self {
        Return {
            pos, value
        }
    }

    #[cfg(test)]
    pub fn return_value(value: Expression) -> Return {
        Return {
            pos: TextPosition::NONEXISTING,
            value: Some(value)
        }
    }

    #[cfg(test)]
    pub fn return_void() -> Return {
        Return {
            pos: TextPosition::NONEXISTING,
            value: None
        }
    }
}