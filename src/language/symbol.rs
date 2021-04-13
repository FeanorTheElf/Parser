use super::ast::*;
use super::identifier::*;
use super::types::*;

// Required to implement the cast which really is no semantical dependency
use super::ast_statement::Statement;

pub trait SymbolDefinitionFuncs: AstNode {

    fn get_name(&self) -> &Name;

    // AFAIK, this is the only way to implement such a cast.
    fn cast_statement_mut(&mut self) -> Option<&mut dyn Statement>;

    fn get_type(&self) -> &Type;
}

dynamic_subtrait!{ SymbolDefinition: SymbolDefinitionFuncs; SymbolDefinitionDynCastable }
