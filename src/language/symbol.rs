use super::super::parser::ast::*;
use super::super::lexer::tokens::Identifier;
use super::super::lexer::error::CompileError;
use super::scope::{ ScopeTable, ScopeData, ScopeNode };
use super::obj_type::Type;

use std::collections::HashMap;

pub struct SymbolInfo<'a> {
    kind: SymbolKind<'a>,
    id_type: Option<Type>,
}

pub enum SymbolKind<'a> {
    LocalVariable(&'a Stmts, &'a Stmt),
    Parameter(&'a Function, &'a ParameterDeclaration),
    StaticFunction(&'a Function)
}

pub type SymbolTable<'a> = HashMap<* const Identifier, SymbolInfo<'a>>;

pub fn annotate_symbols_function<'a>(node: &Function, scopes: &ScopeTable<'a>) {

}
