use super::super::parser::prelude::*;
use super::obj_type::SymbolDefinition;

use std::collections::HashSet;
use std::borrow::Borrow;

pub trait EnumerateDefinitions<'a> {
    type IntoIter: Iterator<Item = &'a dyn SymbolDefinition>;
    fn enumerate(self) -> Self::IntoIter;
}

pub struct GlobalDefinitionsIter<'a> 
{
    iter: std::slice::Iter<'a, FunctionNode>
}

impl<'a> Iterator for GlobalDefinitionsIter<'a>
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|function| function as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a [FunctionNode]
{
    type IntoIter = GlobalDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        GlobalDefinitionsIter {
            iter: self.iter()
        }
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Program
{
    type IntoIter = GlobalDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        self.functions.enumerate()
    }
}

pub struct StmtsNodeDefinitionsIter<'a>
{
    iter: std::slice::Iter<'a, Box<dyn StmtNode>>
}

impl<'a> Iterator for StmtsNodeDefinitionsIter<'a>
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|stmt| stmt.dynamic().downcast_ref::<VariableDeclarationNode>()).map(|decl| decl as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a BlockNode 
{
    type IntoIter = StmtsNodeDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        StmtsNodeDefinitionsIter {
            iter: self.stmts.iter()
        }
    }
}

pub struct ParameterDefinitionsIter<'a>
{
    iter: std::slice::Iter<'a, Box<ParameterNode>>
}

impl<'a> Iterator for ParameterDefinitionsIter<'a>
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|param| &**param as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a FunctionNode 
{
    type IntoIter = ParameterDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        ParameterDefinitionsIter {
            iter: self.params.iter()
        }
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Vec<Box<dyn StmtNode>> 
{
    type IntoIter = StmtsNodeDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        StmtsNodeDefinitionsIter {
            iter: self.iter()
        }
    }
}

#[derive(Clone)]
struct ScopeNode 
{
    definitions: HashSet<Identifier>
}

impl ScopeNode 
{
    fn create<T: ?Sized>(scope: &T) -> ScopeNode 
        where for<'a> &'a T: EnumerateDefinitions<'a>
    {
        ScopeNode {
            definitions: scope.enumerate().map(|def| def.get_ident().clone()).collect()
        }
    }
}

#[derive(Clone)]
pub struct ScopeStack 
{
    scope_stack: Option<Box<ScopeStack>>,
    scope_node: ScopeNode
}

impl ScopeStack 
{
    pub fn new(global: &[FunctionNode]) -> ScopeStack
    {
        ScopeStack {
            scope_stack: None,
            scope_node: ScopeNode::create(global)
        }
    }

    pub fn enter<T>(&mut self, scope: &T)
        where for<'a> &'a T: EnumerateDefinitions<'a>
    {
        let mut result: Box<ScopeStack> = Box::new(ScopeStack 
            {
            scope_stack: std::mem::replace(&mut self.scope_stack, None),
            scope_node: ScopeNode::create(scope)
        });
        std::mem::swap(&mut self.scope_node, &mut result.scope_node);
        self.scope_stack = Some(result);
    }

    pub fn exit(&mut self) 
    {
        let child = std::mem::replace(&mut self.scope_stack, None);
        *self = *child.expect("Cannot call exit() on empty scope stack");
    }

    pub fn is_global(&self) -> bool
    {
        self.scope_stack.is_none()
    }

    pub fn find_definition(&self, identifier: &Identifier) -> Option<&ScopeStack>
    {
        if self.scope_node.definitions.contains(identifier) {
            Some(self)
        } else {
            self.scope_stack.as_ref().and_then(|parent| parent.find_definition(identifier))
        }
    }

    pub fn generate_unique_identifiers(&self, count: usize) -> Vec<Identifier>
    {
        let mut result = Vec::with_capacity(count);
        let mut current = 0;
        for _i in 0..count {
            while self.find_definition(&Identifier::auto(current)).is_some() {
                current = current + 1;
            }
            result.push(Identifier::auto(current));
            current = current + 1;
        }
        return result;
    }

    pub fn rename_disjunct<'a, I>(&'a self, it: I) -> impl 'a + Iterator<Item = (I::Item, Identifier)>
        where I: 'a + Iterator, I::Item: Borrow<Identifier>
    {
        let mut current = 0;
        it.map(move |name: I::Item| {
            if self.find_definition(name.borrow()).is_some() {
                while self.find_definition(&Identifier::auto(current)).is_some() {
                    current = current + 1;
                }
                current = current + 1;
                (name, Identifier::auto(current - 1))
            } else {
                let result = name.borrow().clone();
                (name, result)
            }
        })
    }
}

#[cfg(test)]
impl<'a> EnumerateDefinitions<'a> for &'a Vec<ParameterNode>
{
    type IntoIter = std::iter::Map<std::slice::Iter<'a, ParameterNode>, Box<dyn Fn(&'a ParameterNode) -> &'a dyn SymbolDefinition>>;

    fn enumerate(self) -> Self::IntoIter
    {
        self.iter().map(Box::new(|node: &'a ParameterNode| node as &'a dyn SymbolDefinition))
    }
}
