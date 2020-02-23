use super::super::language::prelude::*;
use super::symbol::SymbolDefinition;

use std::collections::{ HashSet, HashMap };

pub trait EnumerateDefinitions<'a> {
    type IntoIter: Iterator<Item = &'a dyn SymbolDefinition>;
    fn enumerate(self) -> Self::IntoIter;
}

pub struct GlobalDefinitionsIter<'a> 
{
    iter: std::slice::Iter<'a, Box<Function>>
}

impl<'a> Iterator for GlobalDefinitionsIter<'a>
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|function| &**function as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a [Box<Function>]
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
        self.items.enumerate()
    }
}

pub struct BlockDefinitionsIter<'a>
{
    iter: std::slice::Iter<'a, Box<dyn Statement>>
}

impl<'a> Iterator for BlockDefinitionsIter<'a>
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|stmt| stmt.dynamic().downcast_ref::<Declaration>()).map(|decl| decl as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Block 
{
    type IntoIter = BlockDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        BlockDefinitionsIter {
            iter: self.statements.iter()
        }
    }
}

pub struct ParameterDefinitionsIter<'a>
{
    iter: std::slice::Iter<'a, (TextPosition, Name, Type)>
}

impl<'a> Iterator for ParameterDefinitionsIter<'a>
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|param| param as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Function 
{
    type IntoIter = ParameterDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        ParameterDefinitionsIter {
            iter: self.params.iter()
        }
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Vec<Box<dyn Statement>> 
{
    type IntoIter = BlockDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        BlockDefinitionsIter {
            iter: self.iter()
        }
    }
}

#[derive(Clone)]
struct ScopeNode 
{
    definitions: HashSet<Name>
}

impl ScopeNode 
{
    fn create<T: ?Sized>(scope: &T) -> ScopeNode 
        where for<'a> &'a T: EnumerateDefinitions<'a>
    {
        ScopeNode {
            definitions: scope.enumerate().map(|def| def.get_name().clone()).collect()
        }
    }
}

struct ScopeStackIter<'a>
{
    current: Option<&'a ScopeStack<'a>>
}

impl<'a> Iterator for ScopeStackIter<'a>
{
    type Item = &'a ScopeStack<'a>;

    fn next(&mut self) -> Option<Self::Item>
    {
        let result = self.current;
        self.current = self.current.and_then(|scopes| scopes.parent);
        result
    }
}

#[derive(Clone)]
pub struct ScopeStack<'a>
{
    parent: Option<&'a ScopeStack<'a>>,
    scopes: Vec<ScopeNode>
}

impl<'a> ScopeStack<'a>
{
    pub fn new(global: &[Box<Function>]) -> ScopeStack
    {
        ScopeStack {
            parent: None,
            scopes: vec![ScopeNode::create(global)]
        }
    }

    pub fn child_stack<'b>(&'b self) -> ScopeStack<'b>
    {
        ScopeStack {
            parent: Some(self),
            scopes: vec![]
        }
    }

    pub fn enter<T: ?Sized>(&mut self, scope: &T)
        where for<'b> &'b T: EnumerateDefinitions<'b>
    {
        self.scopes.push(ScopeNode::create(scope));
    }

    pub fn exit(&mut self) 
    {
        self.scopes.pop().expect("Cannot call exit() on empty scope stack");
    }

    fn parent_stacks<'b>(&'b self) -> ScopeStackIter<'b>
    {
        ScopeStackIter {
            current: Some(self)
        }
    }

    pub fn definitions<'b>(&'b self) -> impl 'b + Iterator<Item = &'b Name>
    {
        self.parent_stacks().flat_map(|stack| stack.scopes.iter()).flat_map(|scope_node| scope_node.definitions.iter())
    } 

    pub fn rename_disjunct<'b>(&'b self) -> impl 'b + FnMut(Name) -> Name
    {
        let mut current: HashMap<String, u32> = HashMap::new();
        move |name: Name| {
            if let Some(index) = current.get_mut(&name.name) {
                *index += 1;
                Name::new(name.name, *index)
            } else {
                let index: u32 = self.definitions().filter(|def| def.name == name.name).map(|def| def.id).max().unwrap_or(0) + 1;
                current.insert(name.name.clone(), index);
                Name::new(name.name, index)
            }
        }
    }
}

#[cfg(test)]
pub struct StringDefinitionsIter<'a>
{
    iter: std::slice::Iter<'a, Name>
}

#[cfg(test)]
impl<'a> Iterator for StringDefinitionsIter<'a>
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|param| param as &dyn SymbolDefinition)
    }
}

#[cfg(test)]
impl<'a> EnumerateDefinitions<'a> for &'a [Name]
{
    type IntoIter = StringDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        StringDefinitionsIter {
            iter: self.iter()
        }
    }
}