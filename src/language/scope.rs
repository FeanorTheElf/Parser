use super::super::parser::prelude::*;
use super::super::util::ref_eq::{ Ref, RefEq, ref_eq };

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
    parent_scope: Option<&'a dyn Scope>,
    symbol_definitions: Vec<&'a dyn SymbolDefinition>
}

impl<'a> ScopeInfo<'a> {
    pub fn get_definition_of(&self, identifier: &Identifier) -> Option<&'a dyn SymbolDefinition> {
        self.symbol_definitions.iter().find(|def| def.get_identifier() == identifier).map(|def| *def)
    }

    pub fn get_parent_scope(&self) -> Option<&'a dyn Scope> {
        self.parent_scope
    }

    fn add_definition(&mut self, def: &'a dyn SymbolDefinition) -> Result<(), CompileError> {
        if let Some(dupl_def) = self.get_definition_of(def.get_identifier()) {
            return Err(CompileError::new(def.get_annotation().clone(), 
                format!("Duplicate definition of symbol {}, previous definition found at {}", *def.get_identifier(), dupl_def.get_annotation()),
                ErrorType::DuplicateDefinition));
        } else {
            self.symbol_definitions.push(def);
            return Ok(());
        }
    }
}

#[derive(Debug)]
pub struct ScopeTable<'a>(HashMap<Ref<'a, dyn Scope>, ScopeInfo<'a>>);

pub struct DefinedSymbolsIter<'a, 'b, 'c> {
    scopes: &'b ScopeTable<'a>,
    parent_scope: Option<&'c dyn Scope>,
    current_definitions_iter: std::slice::Iter<'b, &'a dyn SymbolDefinition>
}

pub struct ScopeIter<'a, 'b> {
    scopes: &'b ScopeTable<'a>,
    current_scope: Option<&'a dyn Scope>,
}

impl<'a> ScopeTable<'a> {
    pub fn new() -> Self {
        let mut table = HashMap::new();
        table.insert(Ref::from(&GLOBAL as &dyn Scope), ScopeInfo { parent_scope: None, symbol_definitions: vec![] });
        return ScopeTable(table);
    }

    pub fn get<'b>(&'b self, scope: &dyn Scope) -> &'b ScopeInfo<'a> {
        self.0.get(&RefEq::from(scope)).expect("Identifier not found in SymbolTable, did you forget to annotate the syntax tree part?")
    }

    pub fn get_mut<'b>(&'b mut self, scope: & dyn Scope) -> &'b mut ScopeInfo<'a> {
        self.0.get_mut(&RefEq::from(scope)).expect("Identifier not found in SymbolTable, did you forget to annotate the syntax tree part?")
    }

    pub fn visible_symbols_iter<'b, 'c>(&'b self, scope: Option<&'c dyn Scope>) -> DefinedSymbolsIter<'a, 'b, 'c> 
        where 'a: 'c
    {
        if let Some(scope) = scope {
            let scope_info = self.get(scope);
            DefinedSymbolsIter {
                scopes: self,
                parent_scope: scope_info.get_parent_scope(),
                current_definitions_iter: scope_info.symbol_definitions.iter()
            }
        } else {
            DefinedSymbolsIter {
                scopes: self,
                parent_scope: None,
                current_definitions_iter: [].iter()
            }
        }
    }

    pub fn scopes_iter<'b>(&'b self, scope: &'a dyn Scope) -> ScopeIter<'a, 'b> {
        ScopeIter {
            scopes: self,
            current_scope: Some(scope)
        }
    }

    fn insert(&mut self, scope: &'a dyn Scope, data: ScopeInfo<'a>) {
        let old_value = self.0.insert(Ref::from(scope), data);
        assert!(old_value.is_none());
    }
}

pub fn annotate_sope_info_func<'a>(node: &'a FunctionNode, table: &mut ScopeTable<'a>) -> Result<(), CompileError> {
    table.insert(node, ScopeInfo { parent_scope: Some(&GLOBAL), symbol_definitions: vec![] });
    for param in &node.params {
        table.get_mut(node).add_definition(&**param);
    }
    table.get_mut(&GLOBAL).add_definition(node)?;
    match node.implementation.get_concrete() {
        ConcreteFunctionImplementationRef::Implemented(implementation) => {
            annotate_scope_info_stmts(&*implementation.body, table, node)?;
        },
        ConcreteFunctionImplementationRef::Native(native) => { }
    }
    return Ok(());
}

fn annotate_scope_info_stmts<'a>(node: &'a StmtsNode, table: &mut ScopeTable<'a>, parent_scope: &'a dyn Scope) -> Result<(), CompileError> {
    table.insert(node, ScopeInfo { parent_scope: Some(parent_scope), symbol_definitions: vec![] });
    for stmt in &node.stmts {
        annotate_scope_info_stmt(&**stmt, table, node)?;
    }
    return Ok(());
}

fn annotate_scope_info_stmt<'a>(node: &'a dyn StmtNode, table: &mut ScopeTable<'a>, parent_scope: &'a dyn Scope) -> Result<(), CompileError> {
    match node.get_concrete() {
        ConcreteStmtRef::Assignment(_stmt) => { },
        ConcreteStmtRef::Block(stmts) => {
            annotate_scope_info_stmts(stmts, table, parent_scope)?;
        },
        ConcreteStmtRef::Declaration(stmt) => {
            table.get_mut(parent_scope).add_definition(stmt);
        },
        ConcreteStmtRef::Expr(_stmt) => { },
        ConcreteStmtRef::If(stmt) => {
            annotate_scope_info_stmts(&*stmt.block, table, parent_scope)?;
        },
        ConcreteStmtRef::Return(stmt) => { },
        ConcreteStmtRef::While(stmt) => {
            annotate_scope_info_stmts(&*stmt.block, table, parent_scope)?;
        }
    }
    return Ok(());
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

impl<'a, 'b, 'c> Iterator for DefinedSymbolsIter<'a, 'b, 'c> 
    where 'a: 'c
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(definition) = self.current_definitions_iter.next() {
                return Some(*definition);
            } else if let Some(parent_scope) = self.parent_scope {
                let parent_scope_info = self.scopes.get(parent_scope);
                self.parent_scope = parent_scope_info.get_parent_scope();
                self.current_definitions_iter = parent_scope_info.symbol_definitions.iter();
            } else {
                return None;
            }
        }
    }
}

impl<'a, 'b> Iterator for ScopeIter<'a, 'b> 
{
    type Item = (&'a dyn Scope, &'b ScopeInfo<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(scope) = self.current_scope {
            let scope_info = self.scopes.get(scope);
            self.current_scope = scope_info.get_parent_scope();
            Some((scope, scope_info))
        } else {
            None
        }
    }
}
