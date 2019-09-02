use super::super::parser::ast::*;
use super::super::lexer::tokens::Identifier;
use super::super::lexer::error::CompileError;
use super::obj_type::Type;

use std::fmt::Debug;
use std::collections::HashMap;

pub struct GlobalScope();

pub const GLOBAL: GlobalScope = GlobalScope();

pub trait ScopeNode: Debug {
    fn dynamic_cast_global(&self) -> Option<&GlobalScope> {
        None
    }

    fn dynamic_cast_node(&self) -> Option<&dyn Node> {
        None
    }
}

pub trait DeclarationNode: Node + Debug {}

#[derive(Debug)]
pub struct ScopeData<'a> {
    parent_scope: &'a dyn ScopeNode,
    declarations: Vec<&'a dyn DeclarationNode>
}

#[derive(Debug)]
pub struct ScopeTable<'a>(HashMap<* const ScopeNode, ScopeData<'a>>);

impl<'a> ScopeTable<'a> {
    pub fn new() -> Self {
        let mut table = HashMap::new();
        table.insert(&GLOBAL as * const ScopeNode, ScopeData { parent_scope: &GLOBAL, declarations: vec![] });
        return ScopeTable(table);
    }
}

pub fn fill_sope_info_func<'a>(node: &'a Function, table: &mut ScopeTable<'a>) {
    table.0.insert(node as * const ScopeNode, ScopeData { parent_scope: &GLOBAL, declarations: vec![] });
    table.0.get_mut(&(&GLOBAL as * const ScopeNode)).unwrap().declarations.push(node);
    match node {
        Function::Function(ref _annotation, ref _ident, ref _params, ref _type, ref stmts) => {
            fill_scope_info_stmts(stmts, table, node)
        }
    }
}

fn fill_scope_info_stmts<'a>(node: &'a Stmts, table: &mut ScopeTable<'a>, parent_scope: &'a ScopeNode) {
    table.0.insert(node as * const ScopeNode, ScopeData { parent_scope: node, declarations: vec![] });
    match node {
        Stmts::Stmts(ref _annotation, ref stmts) => {
            for stmt in stmts {
            }
        }
    }
}

fn fill_scope_info_stmt<'a>(node: &'a Stmt, table: &mut ScopeTable<'a>, parent_scope: &'a ScopeNode) {
    match node {
        Stmt::Assignment(ref _annotation, ref _assignee, ref _expr) => { 
            unimplemented!();
            //let parent_scope_ptr: * const ScopeNode = parent_scope as * const (dyn ScopeNode + 'static);
            //table.0.get_mut(&parent_scope_ptr).unwrap().declarations.push(node);
        },
        Stmt::Block(ref _annotation, ref stmts) => {
            fill_scope_info_stmts(stmts, table, parent_scope);
        },
        Stmt::Declaration(ref _annotation, ref _type, ref _ident, ref _expr) => { },
        Stmt::Expr(ref _annotation, ref _expr) => { },
        Stmt::If(ref _annotation, ref _cond, ref stmts) => {
            fill_scope_info_stmts(stmts, table, parent_scope);
        },
        Stmt::Return(ref _annotation, ref _expr) => { },
        Stmt::While(ref _annotation, ref _expr, ref stmts) => {
            fill_scope_info_stmts(stmts, table, parent_scope);
        }
    }
}

macro_rules! derive_scope_node {
    ($name:ident) => {
        impl ScopeNode for $name {
            fn dynamic_cast_node(&self) -> Option<&dyn Node> {
                Some(self)
            }
        }
    };
}

derive_scope_node!(Function);
derive_scope_node!(Stmts);

impl Debug for GlobalScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GLOBAL")
    }
}

impl ScopeNode for GlobalScope {
    fn dynamic_cast_global(&self) -> Option<&GlobalScope> {
        Some(self)
    }

    fn dynamic_cast_node(&self) -> Option<&dyn Node> {
        None
    }
}

impl DeclarationNode for Function {}
impl DeclarationNode for Stmt {}