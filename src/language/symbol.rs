use super::super::parser::ast::*;
use super::super::parser::ast_visitor::Visitable;
use super::super::lexer::tokens::Identifier;
use super::super::lexer::error::CompileError;
use super::super::util::ref_eq::{ Ref, RefEq, ref_eq };
use super::scope::{ ScopeTable, ScopeInfo, Scope, SymbolDefinition, GLOBAL, SymbolDefinitionKind };
use super::obj_type::Type;

use std::any::{ Any, TypeId };
use std::collections::HashMap;

#[derive(Debug)]
pub struct SymbolInfo<'a> {
    symbol_definition: &'a dyn SymbolDefinition,
    scope: &'a dyn Scope,
    pub symbol_type: Type,
}

#[derive(Debug)]
pub struct SymbolTable<'a>(HashMap<Ref<'a, Identifier>, SymbolInfo<'a>>);

pub trait SymbolUse : Node {
    fn get_identifier(&self) -> &Identifier;
}

impl<'a> SymbolTable<'a> {
    pub fn new() -> SymbolTable<'a> {
        SymbolTable(HashMap::new())
    }

    pub fn get(&self, ident: &Identifier) -> &SymbolInfo<'a> {
        self.0.get(&RefEq::from(ident)).unwrap()
    }

    fn add_definition(&mut self, definition: &'a dyn SymbolDefinition, definition_scope: &'a dyn Scope, scopes: &ScopeTable<'a>) -> Result<(), CompileError> 
    {
        let ident = definition.get_identifier();

        let parent_scope = scopes.get(definition_scope).get_parent_scope();
        for def in scopes.visible_symbols_iter(parent_scope) {
            if def.get_identifier() == ident {
                return Err(CompileError::new(definition.get_annotation().clone(), 
                    format!("Definition of {} shadows definition found at {}", *ident, def.get_annotation())));
            }
        }
        self.0.insert(Ref::from(definition.get_identifier()), SymbolInfo {
            symbol_definition: definition,
            scope: definition_scope,
            symbol_type: Type::calc_from(definition)?
        });
        return Ok(());
    }

    fn add_use(&mut self, symbol: &'a dyn SymbolUse, use_scope: &'a dyn Scope, scopes: &ScopeTable<'a>) -> Result<(), CompileError> {
        let identifier = symbol.get_identifier();
        let definition = scopes.scopes_iter(use_scope)
            .filter_map(|(scope, scope_info)| scope_info.get_definition_of(identifier).map(|def| (scope, def)))
            .next();

        if let Some((scope, def)) = definition {
            self.0.insert(Ref::from(identifier), SymbolInfo {
                symbol_definition: def,
                scope: scope,
                symbol_type: Type::calc_from(def)?
            });
            return Ok(());
        } else {
            return Err(CompileError::new(symbol.get_annotation().clone(),
                format!("Could not find definition of {}", *identifier)));
        }
    }
}

pub fn annotate_symbols_function<'a>(node: &'a FunctionNode, scopes: &ScopeTable<'a>, symbols: &mut SymbolTable<'a>) -> Result<(), CompileError> {
    symbols.add_definition(node, &GLOBAL, scopes)?;
    for param in &node.params {
        symbols.add_definition(&**param, node, scopes);
    }
    match node.implementation.get_kind() {
        FunctionImplementationKind::Implemented(implementation) => {
            annotate_symbols_stmts(&*implementation.stmts, scopes, symbols)?;
        },
        FunctionImplementationKind::Native(native) => { }
    }
    return Ok(());
}

fn annotate_symbols_stmts<'a>(node: &'a StmtsNode, scopes: &ScopeTable<'a>, symbols: &mut SymbolTable<'a>) -> Result<(), CompileError> {
    for stmt in &node.stmts {
        annotate_symbols_stmt(&**stmt, node, scopes, symbols)?;
    }
    return Ok(());
}

fn annotate_symbols_stmt<'a>(node: &'a dyn StmtNode, parent_scopes: &'a dyn Scope, scopes: &ScopeTable<'a>, symbols: &mut SymbolTable<'a>) -> Result<(), CompileError> {
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

fn annotate_symbols_expr<'a>(node: &'a ExprNode, parent_scope: &'a dyn Scope, scopes: &ScopeTable<'a>, symbols: &mut SymbolTable<'a>) -> Result<(), CompileError> {
    node.iterate(&mut |unary_expr| {
        if let Some(variable_node) = Any::downcast_ref::<VariableNode>(unary_expr.dynamic()) {
            symbols.add_use(variable_node, parent_scope, scopes)?;
        }
        return Ok(());
    })
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