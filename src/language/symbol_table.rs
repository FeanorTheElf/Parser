use super::super::parser::ast::*;
use super::super::lexer::tokens::Identifier;
use super::super::lexer::error::CompileError;
use super::obj_type::Type;

use std::collections::HashMap;

pub type IdentifierKey = i32;

pub struct SymbolInfo {
    kind: SymbolKind,
    id_type: Option<Type>
}

pub enum SymbolKind {
    LocalVariable(*const Stmts, *const Stmt),
    Parameter(*const Function, *const ParameterDeclaration),
    StaticFunction(*const Function)
}

pub struct SymbolTable {
    data: HashMap<IdentifierKey, SymbolInfo>,
    current_key: IdentifierKey
}

impl SymbolTable {

}


fn annotate_scope(node: &mut Function, table: &mut SymbolTable) {

}