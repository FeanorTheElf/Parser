use super::super::parser::ast::*;
use super::super::lexer::tokens::Identifier;
use super::super::lexer::error::CompileError;

use std::iter::FromIterator;
use std::fmt::Debug;
use std::any::Any;
use std::collections::HashMap;

pub struct GlobalScope();

pub static GLOBAL: GlobalScope = GlobalScope();

pub trait Scope: Debug + Any {}

pub trait SymbolDefinition: Node {
    fn get_identifier(&self) -> &Identifier;
    fn get_kind<'a>(&'a self) -> SymbolDefinitionKind<'a>;
}

pub enum SymbolDefinitionKind<'a> {
    LocalVar(&'a VariableDeclarationNode),
    Function(&'a FunctionNode),
    Parameter(&'a ParameterNode)
}

#[derive(Debug)]
pub struct ScopeInfo<'a> {
    parent_scope: &'a dyn Scope,
    symbol_definitions: Vec<&'a dyn SymbolDefinition>
}

impl<'a> ScopeInfo<'a> {
    pub fn get_definition_of(&self, identifier: &Identifier) -> Option<&'a dyn SymbolDefinition> {
        self.symbol_definitions.iter().find(|def| def.get_identifier() == identifier).map(|def| *def)
    } 

    pub fn get_parent_scope(&self) -> &'a dyn Scope {
        self.parent_scope
    }
}

#[derive(Debug)]
pub struct ScopeTable<'a>(HashMap<* const dyn Scope, ScopeInfo<'a>>);

impl<'a> ScopeTable<'a> {
    pub fn new() -> Self {
        let mut table = HashMap::new();
        table.insert(&GLOBAL as * const dyn Scope, ScopeInfo { parent_scope: &GLOBAL, symbol_definitions: vec![] });
        return ScopeTable(table);
    }

    pub fn get<'b>(&'b self, scope: &dyn Scope) -> &'b ScopeInfo<'a> {
        self.0.get(&(scope as * const dyn Scope)).unwrap()
    }

    pub fn get_mut<'b>(&'b mut self, scope: &dyn Scope) -> &'b mut ScopeInfo<'a> {
        self.0.get_mut(&(scope as * const dyn Scope)).unwrap()
    }

    fn insert(&mut self, scope: &dyn Scope, data: ScopeInfo<'a>) {
        let old_value = self.0.insert(scope as * const dyn Scope, data);
        assert!(old_value.is_none());
    }
}

pub fn annotate_sope_info_func<'a>(node: &'a FunctionNode, table: &mut ScopeTable<'a>) {
    table.insert(node, ScopeInfo { 
        parent_scope: &GLOBAL, 
        symbol_definitions: Vec::from_iter(node.params.iter().map(|param| &**param as &'a dyn SymbolDefinition)) 
    });
    table.get_mut(&GLOBAL).symbol_definitions.push(node);
    annotate_scope_info_stmts(&*node.body, table, node)
}

fn annotate_scope_info_stmts<'a>(node: &'a StmtsNode, table: &mut ScopeTable<'a>, parent_scope: &'a dyn Scope) {
    table.insert(node, ScopeInfo { parent_scope: node, symbol_definitions: vec![] });
    for stmt in &node.stmts {
        annotate_scope_info_stmt(&**stmt, table, node);
    }
}

fn annotate_scope_info_stmt<'a>(node: &'a dyn StmtNode, table: &mut ScopeTable<'a>, parent_scope: &'a dyn Scope) {
    match node.get_kind() {
        StmtKind::Assignment(_stmt) => { },
        StmtKind::Block(stmt) => {
            annotate_scope_info_stmts(&*stmt.block, table, parent_scope);
        },
        StmtKind::Declaration(stmt) => {
            table.get_mut(parent_scope).symbol_definitions.push(stmt);
        },
        StmtKind::Expr(_stmt) => { },
        StmtKind::If(stmt) => {
            annotate_scope_info_stmts(&*stmt.block, table, parent_scope);
        },
        StmtKind::Return(stmt) => { },
        StmtKind::While(stmt) => {
            annotate_scope_info_stmts(&*stmt.block, table, parent_scope);
        }
    }
}

impl Scope for FunctionNode {}
impl Scope for StmtsNode {}

impl Debug for GlobalScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GLOBAL")
    }
}

impl Scope for GlobalScope {}

impl SymbolDefinition for FunctionNode {
    fn get_identifier(&self) -> &Identifier {
        &self.ident
    }

    fn get_kind<'a>(&'a self) -> SymbolDefinitionKind<'a> {
        SymbolDefinitionKind::Function(&self)
    }
}
impl SymbolDefinition for VariableDeclarationNode {
    fn get_identifier(&self) -> &Identifier {
        &self.ident
    }

    fn get_kind<'a>(&'a self) -> SymbolDefinitionKind<'a> {
        SymbolDefinitionKind::LocalVar(&self)
    }
}
impl SymbolDefinition for ParameterNode {
    fn get_identifier(&self) -> &Identifier {
        &self.ident
    }

    fn get_kind<'a>(&'a self) -> SymbolDefinitionKind<'a> {
        SymbolDefinitionKind::Parameter(&self)
    }
}