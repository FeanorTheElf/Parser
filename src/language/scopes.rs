use super::identifier::Name;
use super::position::TextPosition;
use super::error::CompileError;
use super::symbol::SymbolDefinition;
use std::collections::HashMap;

pub type DefinitionScopeStackConst<'a, 'b> = ScopeStack<'a, &'b dyn SymbolDefinition>;
pub type DefinitionScopeStackMut<'a, 'b> = ScopeStack<'a, &'b mut dyn SymbolDefinition>;

pub trait DefinitionEnvironment<'a, 'b>: std::fmt::Debug {

    fn get<'c>(&'c self, name: &Name) -> Option<&'c dyn SymbolDefinition>;

    fn get_defined<'c>(&'c self, name: &Name, pos: &TextPosition) -> Result<&'c dyn SymbolDefinition, CompileError> {
        self.get(name).ok_or_else(|| {
            CompileError::undefined_symbol(name, pos)
        })
    }
}

impl<'a, 'b> DefinitionEnvironment<'a, 'b> for DefinitionScopeStackMut<'a, 'b> {
    
    fn get<'c>(&'c self, name: &Name) -> Option<&'c dyn SymbolDefinition> {
        self.get(name).map(|x| *x as &dyn SymbolDefinition)
    }
}

impl<'a, 'b> DefinitionEnvironment<'a, 'b> for DefinitionScopeStackConst<'a, 'b> {
    
    fn get<'c>(&'c self, name: &Name) -> Option<&'c dyn SymbolDefinition> {
        self.get(name).map(|x| *x)
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
    definitions: HashMap<Name, T>,
}

impl<'a, T> ScopeStack<'a, T> {

    pub fn new() -> ScopeStack<'a, T> {
        ScopeStack {
            parent: None,
            definitions: HashMap::new(),
        }
    }

    pub fn child_stack<'b>(&'b self) -> ScopeStack<'b, T> {
        ScopeStack {
            parent: Some(self),
            definitions: HashMap::new(),
        }
    }

    fn all_stacks<'b>(&'b self) -> ScopeStackIter<'b, T> {
        ScopeStackIter {
            current: Some(self),
        }
    }

    pub fn is_global_scope<'b>(self: &'b Self) -> bool {
        self.parent.is_none()
    }

    fn this_scope_definitions<'b>(&'b self) -> impl 'b + Iterator<Item = (&'b Name, &'b T)> {
        self.definitions.iter()
    }

    pub fn definitions<'b>(&'b self) -> impl 'b + Iterator<Item = (&'b Name, &'b T)> {
        self.all_stacks().flat_map(|stack| stack.this_scope_definitions())
    }

    pub fn rename_disjunct<'b>(&'b self, mut name: Name) -> Name {
        while self.get(&name).is_some() {
            name.id += 1;
        }
        return name;
    }

    pub fn get<'b>(&'b self, name: &Name) -> Option<&'b T> {
        self.all_stacks().find_map(|stack| {
            stack.definitions.get(name)
        })
    }

    pub fn get_defined<'b>(&'b self, name: &Name, pos: &TextPosition) -> Result<&'b T, CompileError> {
        self.get(name).ok_or_else(|| {
            CompileError::undefined_symbol(name, pos)
        })
    }

    pub fn get_global_scope(&self) -> &ScopeStack<'a, T> {
        let mut result = self;
        while let Some(parent) = result.parent {
            result = parent;
        }
        return result;
    }

    pub fn register(&mut self, name: Name, val: T) {
        let old = self.definitions.insert(name, val);
        assert!(old.is_none());
    }

    pub fn unregister(&mut self, name: &Name) -> T {
        let entry = self.definitions.remove(name);
        assert!(entry.is_some());
        return entry.unwrap();
    }
}

impl<'a, 'b> ScopeStack<'a, &'b dyn SymbolDefinition> {

    pub fn register_symbol(&mut self, symbol: &'b dyn SymbolDefinition) {
        self.register(symbol.get_name().clone(), symbol)
    }
}

impl<'a, 'b> ScopeStack<'a, &'b mut dyn SymbolDefinition> {

    pub fn register_symbol(&mut self, symbol: &'b mut dyn SymbolDefinition) {
        self.register(symbol.get_name().clone(), symbol)
    }
}