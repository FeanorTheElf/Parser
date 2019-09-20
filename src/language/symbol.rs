use super::super::parser::ast::*;
use super::super::lexer::tokens::Identifier;
use super::super::lexer::error::CompileError;
use super::scope::{ ScopeTable, ScopeInfo, Scope, SymbolDefinition, GLOBAL, SymbolDefinitionKind };
use super::obj_type::Type;

use std::collections::HashMap;

pub struct SymbolInfo<'a> {
    symbol_definition: &'a dyn SymbolDefinition,
    scope: &'a dyn Scope,
    symbol_type: Type,
}

pub struct SymbolTable<'a>(HashMap<* const Identifier, SymbolInfo<'a>>);

pub trait SymbolUse : Node {
    fn get_identifier(&self) -> &Identifier;
}

impl<'a> SymbolTable<'a> {
    pub fn new() -> SymbolTable<'a> {
        SymbolTable(HashMap::new())
    }

    fn add_definition(&mut self, definition: &'a dyn SymbolDefinition, scope: &'a dyn Scope, scopes: &ScopeTable<'a>) -> Result<(), CompileError> {
        let ident = definition.get_identifier();
        if let Some(duplicate_def) = scopes.get(scope).get_definition_of(ident) {
            return Err(CompileError::new(definition.get_annotation().clone(), 
                format!("Duplicate definition of symbol {}, other definition found at {}", *ident, duplicate_def.get_annotation())));
        } else {
            let mut current_scope = scopes.get(scope).get_parent_scope();
            while !std::ptr::eq(current_scope, &GLOBAL) {
                if let Some(duplicate_def) = scopes.get(current_scope).get_definition_of(ident) {
                    return Err(CompileError::new(definition.get_annotation().clone(), 
                        format!("Definition of {} shadows definition found at {}", *ident, duplicate_def.get_annotation())));
                }
                current_scope = scopes.get(current_scope).get_parent_scope();
            }
            self.0.insert(definition.get_identifier() as * const Identifier, SymbolInfo {
                symbol_definition: definition,
                scope: scope,
                symbol_type: Type::calc_from(definition)?
            });
            return Ok(());
        }
    }

    fn add_use(&mut self, symbol: &'a dyn SymbolUse, scope: &'a dyn Scope, scopes: &ScopeTable<'a>) -> Result<(), CompileError> {
        let ident = symbol.get_identifier();
        let mut current_scope = scope;
        while !std::ptr::eq(current_scope, &GLOBAL) && scopes.get(current_scope).get_definition_of(ident).is_none() {
            current_scope = scopes.get(current_scope).get_parent_scope();
        }
        if let Some(definition) = scopes.get(current_scope).get_definition_of(ident) {
            self.0.insert(ident as * const Identifier, SymbolInfo {
                symbol_definition: definition,
                scope: scope,
                symbol_type: Type::calc_from(definition)?
            });
            return Ok(());
        } else {
            return Err(CompileError::new(symbol.get_annotation().clone(),
                format!("Could not find definition of {}", *ident)));
        }
    }
}

pub fn annotate_symbols_function<'a>(node: &'a FunctionNode, scopes: &ScopeTable<'a>, symbols: &mut SymbolTable<'a>) -> Result<(), CompileError> {
    symbols.add_definition(node, &GLOBAL, scopes)?;
    for param in &node.params {
        symbols.add_definition(&**param, node, scopes);
    }
    annotate_symbols_stmts(&*node.body, scopes, symbols)?;
    return Ok(());
}

pub fn annotate_symbols_stmts<'a>(node: &'a StmtsNode, scopes: &ScopeTable<'a>, symbols: &mut SymbolTable<'a>) -> Result<(), CompileError> {
    for stmt in &node.stmts {
        annotate_symbols_stmt(&**stmt, node, scopes, symbols)?;
    }
    return Ok(());
}

pub fn annotate_symbols_stmt<'a>(node: &'a dyn StmtNode, parent_scopes: &'a dyn Scope, scopes: &ScopeTable<'a>, symbols: &mut SymbolTable<'a>) -> Result<(), CompileError> {
    match node.get_kind() {
        StmtKind::Assignment(stmt) => {
            annotate_symbols_expr(&*stmt.assignee, parent_scopes, scopes, symbols)?;
            annotate_symbols_expr(&*stmt.expr, parent_scopes, scopes, symbols)?;
        },
        StmtKind::Block(stmt) => {
            annotate_symbols_stmts(&*stmt.block, scopes, symbols)?;
        },
        StmtKind::Declaration(stmt) => {
            symbols.add_definition(stmt, parent_scopes, scopes)?;
            annotate_symbols_expr(&*stmt.expr, parent_scopes, scopes, symbols)?;
        },
        StmtKind::Expr(stmt) => {
            annotate_symbols_expr(&*stmt.expr, parent_scopes, scopes, symbols)?;
        },
        StmtKind::If(stmt) => {
            annotate_symbols_expr(&*stmt.condition, parent_scopes, scopes, symbols)?;
            annotate_symbols_stmts(&*stmt.block, scopes, symbols)?;
        },
        StmtKind::While(stmt) => {
            annotate_symbols_expr(&*stmt.condition, parent_scopes, scopes, symbols)?;
            annotate_symbols_stmts(&*stmt.block, scopes, symbols)?;
        },
        StmtKind::Return(stmt) => {
            annotate_symbols_expr(&*stmt.expr, parent_scopes, scopes, symbols)?;
        }
    }
    return Ok(());
}

pub fn annotate_symbols_expr<'a>(node: &'a ExprNode, parent_scope: &'a dyn Scope, scopes: &ScopeTable<'a>, symbols: &mut SymbolTable<'a>) -> Result<(), CompileError> {
    
    return Ok(());
}

impl SymbolUse for VariableNode {
    fn get_identifier(&self) -> &Identifier {
        &self.identifier
    }
}

impl SymbolUse for FunctionCallNode {
    fn get_identifier(&self) -> &Identifier {
        &self.function
    }
}