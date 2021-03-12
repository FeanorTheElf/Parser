use super::ast::SymbolDefinition;
use super::identifier::Name;
use super::position::TextPosition;
use super::error::{CompileError, ErrorType};
use std::collections::HashMap;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
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
    fn new() -> ScopeNode<T> {
        ScopeNode { definitions: HashMap::new() }
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
    scopes: ScopeNode<T>,
}

impl<'a, T> ScopeStack<'a, T> {
    pub fn new() -> ScopeStack<'a, T> {
        ScopeStack {
            parent: None,
            scopes: ScopeNode::new(),
        }
    }

    pub fn child_stack<'b>(&'b self) -> ScopeStack<'b, T> {
        ScopeStack {
            parent: Some(self),
            scopes: ScopeNode::new(),
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
        self.scopes.definitions.iter()
    }

    pub fn definitions<'b>(&'b self) -> impl 'b + Iterator<Item = (&'b Name, &'b T)> {
        self.all_stacks().flat_map(|stack| stack.this_scope_definitions())
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
            stack.scopes.definitions.get(name)
        })
    }

    pub fn get_defined(&self, name: &Name, pos: &TextPosition) -> Result<&T, CompileError> {
        self.get(name).ok_or_else(|| {
            CompileError::undefined_symbol(name, pos)
        })
    }
}
