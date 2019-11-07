use super::super::parser::prelude::*;
use super::super::parser::ast_visitor::Visitable;
use super::super::util::ref_eq::{ Ref, RefEq };
use super::scope::{ ScopeTable, ScopeInfo, Scope, SymbolDefinition, GLOBAL, SymbolDefinitionKind };
use super::obj_type::Type;

#[cfg(test)]
use super::scope::annotate_sope_info_func;
#[cfg(test)]
use super::obj_type::PrimitiveType;
#[cfg(test)]
use super::super::parser::Parse;
#[cfg(test)]
use super::super::lexer::lexer::lex;

use std::any::{ Any, TypeId };
use std::collections::HashMap;

#[derive(Debug)]
pub enum SymbolInfo<'a> {
    Definition(SymbolDefinitionInfo<'a>),
    Use(SymbolUseInfo<'a>)
}

#[derive(Debug)]
pub struct SymbolDefinitionInfo<'a> {
    symbol_type: Type,
    scope: &'a dyn Scope,
    uses: Vec<&'a dyn SymbolUse>,
    pub definition_node: &'a dyn SymbolDefinition
}

#[derive(Debug)]
pub struct SymbolUseInfo<'a> {
    symbol_definition: &'a dyn SymbolDefinition,
    use_node: &'a dyn SymbolUse
}

impl<'a> SymbolInfo<'a> {
    pub fn expect_definition(&self) -> &SymbolDefinitionInfo<'a> {
        match self {
            SymbolInfo::Definition(ref definition) => definition,
            SymbolInfo::Use(ref _reference) => panic!("Expected symbol definition, got reference")
        }
    }

    pub fn expect_use(&self) -> &SymbolUseInfo<'a> {
        match self {
            SymbolInfo::Definition(ref _definition) => panic!("Expected symbol definition, got reference"),
            SymbolInfo::Use(ref reference) => reference
        }
    }
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

    pub fn get_type(&self, identifier: &Identifier) -> &Type {
        &self.get_identifier_definition(identifier).symbol_type
    }

    pub fn get_identifier_definition(&self, identifier: &Identifier) -> &SymbolDefinitionInfo<'a> {
        match self.get(identifier) {
            SymbolInfo::Definition(ref definition) => &definition,
            SymbolInfo::Use(ref reference) => &self.get_definition(reference)
        }
    }

    fn get_definition(&self, info: &SymbolUseInfo<'a>) -> &SymbolDefinitionInfo<'a> {
        self.get(info.symbol_definition.get_identifier()).expect_definition()
    }

    pub fn get(&self, ident: &Identifier) -> &SymbolInfo<'a> {
        self.0.get(&RefEq::from(ident)).expect("Identifier not found in SymbolTable, did you forget to annotate the syntax tree part?")
    }

    fn get_mut(&mut self, ident: &Identifier) -> &mut SymbolInfo<'a> {
        self.0.get_mut(&RefEq::from(ident)).expect("Identifier not found in SymbolTable, did you forget to annotate the syntax tree part?")
    }

    fn ensure_not_shadowed(&self, definition: &'a dyn SymbolDefinition, definition_scope: &'a dyn Scope, scopes: &ScopeTable<'a>) -> Result<(), CompileError> 
    {
        let ident = definition.get_identifier();
        let parent_scope = scopes.get(definition_scope).get_parent_scope();
        for def in scopes.visible_symbols_iter(parent_scope) {
            if def.get_identifier() == ident {
                return Err(CompileError::new(definition.get_annotation().clone(), 
                    format!("Definition of {} shadows definition found at {}", *ident, def.get_annotation()),
                    ErrorType::ShadowedDefinition));
            }
        }
        return Ok(());
    }

    fn add_symbol_definition(&mut self, definition: &'a dyn SymbolDefinition, definition_scope: &'a dyn Scope, scopes: &ScopeTable<'a>) -> Result<(), CompileError> 
    {
        self.ensure_not_shadowed(definition, definition_scope, scopes)?;
        if !self.0.contains_key(&RefEq::from(definition.get_identifier())) {
            self.0.insert(Ref::from(definition.get_identifier()), SymbolInfo::Definition(SymbolDefinitionInfo {
                symbol_type: Type::calc_from(definition)?,
                scope: definition_scope,
                uses: vec![],
                definition_node: definition
            }));
        }
        return Ok(());
    }

    fn add_use_to_definition(&mut self, symbol: &'a dyn SymbolUse, def: &'a dyn SymbolDefinition, def_scope: &'a dyn Scope) -> Result<(), CompileError> {
        if let Some(definition_info) = self.0.get_mut(&RefEq::from(def.get_identifier())) {
            match definition_info {
                SymbolInfo::Definition(ref mut info) => {
                    info.uses.push(symbol);
                },
                SymbolInfo::Use(ref _reference) => panic!("") 
            }
        } else {
            self.0.insert(Ref::from(def.get_identifier()), SymbolInfo::Definition(SymbolDefinitionInfo {
                symbol_type: Type::calc_from(def)?,
                scope: def_scope,
                uses: vec![symbol],
                definition_node: def
            }));
        }
        return Ok(());
    }

    fn add_symbol_use(&mut self, symbol: &'a dyn SymbolUse, use_scope: &'a dyn Scope, scopes: &ScopeTable<'a>) -> Result<(), CompileError> 
    {
        let identifier = symbol.get_identifier();
        let definition = scopes.scopes_iter(use_scope)
            .filter_map(|(scope, scope_info)| scope_info.get_definition_of(identifier).map(|def| (scope, def)))
            .next();

        if let Some((def_scope, def)) = definition {
            self.add_use_to_definition(symbol, def, def_scope)?;
            self.0.insert(Ref::from(identifier), SymbolInfo::Use(SymbolUseInfo {
                symbol_definition: def,
                use_node: symbol
            }));
            return Ok(());
        } else {
            return Err(CompileError::new(symbol.get_annotation().clone(),
                format!("Could not find definition of {}", *identifier),
                ErrorType::UndefinedSymbol));
        }
    }
}

pub fn annotate_symbols_function<'a>(node: &'a FunctionNode, scopes: &ScopeTable<'a>, symbols: &mut SymbolTable<'a>) -> Result<(), CompileError> {
    symbols.add_symbol_definition(node, &GLOBAL, scopes)?;
    for param in &node.params {
        symbols.add_symbol_definition(&**param, node, scopes);
    }
    match node.implementation.get_concrete() {
        ConcreteFunctionImplementationRef::Implemented(implementation) => {
            annotate_symbols_stmts(&*implementation.body, scopes, symbols)?;
        },
        ConcreteFunctionImplementationRef::Native(native) => { }
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
    match node.get_concrete() {
        ConcreteStmtRef::Assignment(stmt) => {
            annotate_symbols_expr(&*stmt.assignee, parent_scopes, scopes, symbols)?;
            annotate_symbols_expr(&*stmt.expr, parent_scopes, scopes, symbols)?;
        },
        ConcreteStmtRef::Block(stmts) => {
            annotate_symbols_stmts(stmts, scopes, symbols)?;
        },
        ConcreteStmtRef::Declaration(stmt) => {
            symbols.add_symbol_definition(stmt, parent_scopes, scopes)?;
            annotate_symbols_expr(&*stmt.expr, parent_scopes, scopes, symbols)?;
        },
        ConcreteStmtRef::Expr(stmt) => {
            annotate_symbols_expr(&*stmt.expr, parent_scopes, scopes, symbols)?;
        },
        ConcreteStmtRef::If(stmt) => {
            annotate_symbols_expr(&*stmt.condition, parent_scopes, scopes, symbols)?;
            annotate_symbols_stmts(&*stmt.block, scopes, symbols)?;
        },
        ConcreteStmtRef::While(stmt) => {
            annotate_symbols_expr(&*stmt.condition, parent_scopes, scopes, symbols)?;
            annotate_symbols_stmts(&*stmt.block, scopes, symbols)?;
        },
        ConcreteStmtRef::Return(stmt) => {
            annotate_symbols_expr(&*stmt.expr, parent_scopes, scopes, symbols)?;
        }
    }
    return Ok(());
}

fn annotate_symbols_expr<'a>(node: &'a ExprNode, parent_scope: &'a dyn Scope, scopes: &ScopeTable<'a>, symbols: &mut SymbolTable<'a>) -> Result<(), CompileError> {
    node.iterate(&mut |unary_expr| {
        if let Some(variable_node) = unary_expr.dynamic().downcast_ref::<VariableNode>() {
            symbols.add_symbol_use(variable_node, parent_scope, scopes)?;
        } else if let Some(function_call_node) = unary_expr.dynamic().downcast_ref::<FunctionCallNode>() {
            symbols.add_symbol_use(function_call_node, parent_scope, scopes);
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

#[test]
fn test_correct_definitions() {
    let len = FunctionNode::parse(&mut lex("fn len(a: int[],): int { let b: int[] = a; { return len(b); } }")).unwrap();

    let mut scopes = ScopeTable::new();
    assert!(annotate_sope_info_func(&len, &mut scopes).is_ok());

    let mut symbols = SymbolTable::new();
    assert!(annotate_symbols_function(&len, &scopes, &mut symbols).is_ok());

    assert_eq!(&Type::Function(vec![Type::Array(PrimitiveType::Int, 1)], Some(Box::new(Type::Primitive(PrimitiveType::Int)))), symbols.get_type(&len.ident));

    let a_use: &VariableNode = len.implementation.dynamic().downcast_ref::<ImplementedFunctionNode>().unwrap().body.stmts[0].dynamic()
        .downcast_ref::<VariableDeclarationNode>().unwrap().expr.head.head.head.head.head.head.dynamic().downcast_ref::<VariableNode>().unwrap();
    assert_eq!(vec![ Ref::from(&a_use.identifier) ], 
        symbols.get(&len.params[0].ident).expect_definition().uses.iter().map(|var| Ref::from(var.get_identifier())).collect::<Vec<Ref<Identifier>>>());

    let len_use: &FunctionCallNode = len.implementation.dynamic().downcast_ref::<ImplementedFunctionNode>().unwrap().body.stmts[1].dynamic()
        .downcast_ref::<StmtsNode>().unwrap().stmts[0].dynamic().downcast_ref::<ReturnNode>().unwrap().expr.head.head.head.head.head.head
        .dynamic().downcast_ref::<FunctionCallNode>().unwrap();
    assert_eq!(vec![ Ref::from(&len_use.function) ], symbols.get(&len.ident).expect_definition().uses.iter().map(|var| Ref::from(var.get_identifier())).collect::<Vec<Ref<Identifier>>>());
}

#[test]
fn test_definition_not_found() {
    let function = FunctionNode::parse(&mut lex("fn test(): void { let b: int = a; }")).unwrap();

    let mut scopes = ScopeTable::new();
    assert!(annotate_sope_info_func(&function, &mut scopes).is_ok());

    let mut symbols = SymbolTable::new();
    assert_eq!(&TextPosition::create(0, 31), annotate_symbols_function(&function, &scopes, &mut symbols).unwrap_err().get_position());
}

#[test]
fn test_definition_shadowed() {
    let function = FunctionNode::parse(&mut lex("fn b(a: int,): void { let b: int = a; }")).unwrap();

    let mut scopes = ScopeTable::new();
    assert!(annotate_sope_info_func(&function, &mut scopes).is_ok());

    let mut symbols = SymbolTable::new();
    assert_eq!(&TextPosition::create(0, 22), annotate_symbols_function(&function, &scopes, &mut symbols).unwrap_err().get_position());
}