use super::ast::*;
use super::identifier::*;
use super::types::*;
use super::position::TextPosition;

// Required to implement the cast which really is no semantical dependency
use super::ast_statement::Statement;

pub trait SymbolDefinitionFuncs: AstNode {

    fn get_name(&self) -> &Name;

    fn get_name_mut(&mut self) -> &mut Name;

    // AFAIK, this is the only way to implement such a cast.
    fn cast_statement_mut(&mut self) -> Option<&mut dyn Statement>;

    fn get_type(&self) -> &Type;
}

dynamic_subtrait!{ SymbolDefinition: SymbolDefinitionFuncs; SymbolDefinitionDynCastable }

#[derive(Debug, PartialEq, Eq)]
pub struct DummySymbolDefinition;

impl AstNodeFuncs for DummySymbolDefinition {

    fn pos(&self) -> &TextPosition {
        panic!("DummySymbolDefinition does not provide functionality")
    }
}
impl AstNode for DummySymbolDefinition {}

impl SymbolDefinitionFuncs for DummySymbolDefinition {

    fn get_name(&self) -> &Name {
        panic!("DummySymbolDefinition does not provide functionality")
    }

    fn get_name_mut(&mut self) -> &mut Name {
        panic!("DummySymbolDefinition does not provide functionality")
    }

    fn cast_statement_mut(&mut self) -> Option<&mut dyn Statement> {
        None
    }

    fn get_type(&self) -> &Type {
        panic!("DummySymbolDefinition does not provide functionality")
    }
}

impl SymbolDefinition for DummySymbolDefinition {}

pub const DUMMY_SYMBOL_DEFINITION: DummySymbolDefinition = DummySymbolDefinition;