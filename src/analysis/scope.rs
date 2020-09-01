use super::super::language::prelude::*;
use super::symbol::SymbolDefinition;
use std::collections::HashMap;

pub trait EnumerateDefinitions<'a> {
    type IntoIter: Iterator<Item = &'a dyn SymbolDefinition>;

    fn enumerate(self) -> Self::IntoIter;
}

pub struct GlobalDefinitionsIter<'a> {
    iter: std::slice::Iter<'a, Box<Function>>,
}

impl<'a> Iterator for GlobalDefinitionsIter<'a> {
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {

        self.iter
            .next()
            .map(|function| &**function as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a [Box<Function>] {
    type IntoIter = GlobalDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter {

        GlobalDefinitionsIter { iter: self.iter() }
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Program {
    type IntoIter = GlobalDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter {

        self.items.enumerate()
    }
}

pub struct BlockDefinitionsIter<'a> {
    iter: std::slice::Iter<'a, Box<dyn Statement>>,
}

impl<'a> Iterator for BlockDefinitionsIter<'a> {
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {

        self.iter.find_map(|stmt| {

            stmt.dynamic()
                .downcast_ref::<LocalVariableDeclaration>()
                .map(|decl| &decl.declaration as &dyn SymbolDefinition)
                .or(stmt
                    .dynamic()
                    .downcast_ref::<Label>()
                    .map(|decl| decl as &dyn SymbolDefinition))
        })
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Block {
    type IntoIter = BlockDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter {

        BlockDefinitionsIter {
            iter: self.statements.iter(),
        }
    }
}

pub struct ParameterDefinitionsIter<'a> {
    iter: std::slice::Iter<'a, Declaration>,
}

impl<'a> Iterator for ParameterDefinitionsIter<'a> {
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {

        self.iter.next().map(|param| param as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Function {
    type IntoIter = ParameterDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter {

        ParameterDefinitionsIter {
            iter: self.params.iter(),
        }
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a ParallelFor {
    type IntoIter = ParameterDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter {

        ParameterDefinitionsIter {
            iter: self.index_variables.iter(),
        }
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Vec<Box<dyn Statement>> {
    type IntoIter = BlockDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter {

        BlockDefinitionsIter { iter: self.iter() }
    }
}

#[cfg(test)]

pub struct TupleDefinitionsIter<'a> {
    iter: std::slice::Iter<'a, (Name, Type)>,
}

#[cfg(test)]

impl<'a> Iterator for TupleDefinitionsIter<'a> {
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {

        self.iter.next().map(|param| param as &dyn SymbolDefinition)
    }
}

#[cfg(test)]

impl<'a> EnumerateDefinitions<'a> for &'a [(Name, Type)] {
    type IntoIter = TupleDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter {

        TupleDefinitionsIter { iter: self.iter() }
    }
}

pub struct NoData;

impl<'a> From<&'a dyn SymbolDefinition> for NoData {
    fn from(_: &'a dyn SymbolDefinition) -> Self {

        NoData
    }
}

pub type NameScopeStack<'a> = ScopeStack<'a, NoData>;

pub type DefinitionScopeStack<'a, 'b> = ScopeStack<'a, &'b dyn SymbolDefinition>;

#[derive(Debug, Clone)]

struct ScopeNode<T> {
    definitions: HashMap<Name, T>,
}

impl<T> ScopeNode<T> {
    fn create<'c, 'a, S: ?Sized>(scope: &'c S) -> ScopeNode<T>
    where
        &'c S: EnumerateDefinitions<'a>,
        T: From<&'a dyn SymbolDefinition>,
    {

        let defs = scope
            .enumerate()
            .map(|def| (def.get_name().clone(), T::from(def)))
            .collect();

        ScopeNode { definitions: defs }
    }
}

struct ScopeStackIter<'a, T> {
    current: Option<&'a ScopeStack<'a, T>>,
}

impl<'a, T> Iterator for ScopeStackIter<'a, T> {
    type Item = &'a ScopeStack<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {

        let result = self.current;

        self.current = self.current.and_then(|scopes| scopes.parent);

        result
    }
}

#[derive(Debug, Clone)]

pub struct ScopeStack<'a, T> {
    parent: Option<&'a ScopeStack<'a, T>>,
    scopes: Vec<ScopeNode<T>>,
}

impl<'a, T> ScopeStack<'a, T> {
    pub fn new<'b>(global: &'b [Box<Function>]) -> ScopeStack<'a, T>
    where
        T: From<&'b dyn SymbolDefinition>,
    {

        ScopeStack {
            parent: None,
            scopes: vec![ScopeNode::create(global)],
        }
    }

    pub fn child_stack<'b>(&'b self) -> ScopeStack<'b, T> {

        ScopeStack {
            parent: Some(self),
            scopes: vec![],
        }
    }

    pub fn child_scope<'b, 'c, S: ?Sized>(&'b self, scope: &'c S) -> ScopeStack<'b, T>
    where
        &'c S: EnumerateDefinitions<'c>,
        T: From<&'c dyn SymbolDefinition>,
    {

        let mut result = self.child_stack();

        result.enter(scope);

        return result;
    }

    pub fn enter<'c, 'b, S: ?Sized>(&mut self, scope: &'c S)
    where
        &'c S: EnumerateDefinitions<'b>,
        T: From<&'b dyn SymbolDefinition>,
    {

        self.scopes.push(ScopeNode::create(scope));
    }

    pub fn exit(&mut self) {

        self.scopes
            .pop()
            .expect("Cannot call exit() on empty scope stack");
    }

    fn all_stacks<'b>(&'b self) -> ScopeStackIter<'b, T> {

        ScopeStackIter {
            current: Some(self),
        }
    }

    pub fn is_global_scope<'b>(self: &'b Self) -> bool {

        self.parent.is_none()
    }

    fn non_global_stacks<'b>(&'b self) -> impl 'b + Iterator<Item = &'b ScopeStack<'b, T>> {

        self.all_stacks().filter(|stack| !stack.is_global_scope())
    }

    pub fn definitions<'b>(&'b self) -> impl 'b + Iterator<Item = (&'b Name, &'b T)> {

        self.all_stacks()
            .flat_map(|stack| stack.scopes.iter())
            .flat_map(|scope_node| scope_node.definitions.iter())
    }

    pub fn non_global_definitions<'b>(&'b self) -> impl 'b + Iterator<Item = (&'b Name, &'b T)> {

        self.non_global_stacks()
            .flat_map(|stack| stack.scopes.iter())
            .flat_map(|scope_node| scope_node.definitions.iter())
    }

    pub fn rename_disjunct<'b>(&'b self) -> impl 'b + FnMut(Name) -> Name {

        let mut current: HashMap<String, u32> = HashMap::new();

        move |name: Name| {
            if let Some(index) = current.get_mut(&name.name) {

                *index += 1;

                Name::new(name.name, *index)
            } else {

                let index: u32 = self
                    .definitions()
                    .filter(|def| def.0.name == name.name)
                    .map(|def| def.0.id)
                    .max()
                    .map(|x| x + 1)
                    .unwrap_or(0);

                current.insert(name.name.clone(), index);

                Name::new(name.name, index)
            }
        }
    }

    pub fn get(&self, name: &Name) -> Option<&T> {

        self.all_stacks().find_map(|stack| {

            stack
                .scopes
                .iter()
                .find_map(|scope| scope.definitions.get(name))
        })
    }

    pub fn get_defined(&self, name: &Name, pos: &TextPosition) -> Result<&T, CompileError> {

        self.get(name).ok_or_else(|| {

            CompileError::new(
                pos,
                format!("Undefined symbol {}", name),
                ErrorType::UndefinedSymbol,
            )
        })
    }
}

#[cfg(test)]

pub struct StringDefinitionsIter<'a> {
    iter: std::slice::Iter<'a, Name>,
}

#[cfg(test)]

impl<'a> Iterator for StringDefinitionsIter<'a> {
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {

        self.iter.next().map(|param| param as &dyn SymbolDefinition)
    }
}

#[cfg(test)]

impl<'a> EnumerateDefinitions<'a> for &'a [Name] {
    type IntoIter = StringDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter {

        StringDefinitionsIter { iter: self.iter() }
    }
}
