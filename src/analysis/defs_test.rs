use super::super::language::prelude::*;
use super::symbol::SymbolDefinition;
use super::scope::EnumerateDefinitions;

pub struct EnvironmentBuilder {
    types: TypeVec,
    defs: Definitions
}

impl EnvironmentBuilder {

    pub fn new() -> Self {
        EnvironmentBuilder {
            types: TypeVec::new(),
            defs: Definitions {
                defs: Vec::new()
            }
        }
    }

    pub fn types(&mut self) -> &mut TypeVec {
        &mut self.types
    }

    pub fn add_array_def(mut self, name: &str, base: PrimitiveType, dimension_count: usize) -> Self {
        let ty = self.types.get_array_type(base, dimension_count, true);
        self.defs.defs.push(Box::new((Name::l(name), ty)));
        return self;
    }

    pub fn add_view_def(mut self, name: &str, base: PrimitiveType, dimension_count: usize) -> Self {
        let ty = self.types.get_view_type(base, dimension_count, true);
        self.defs.defs.push(Box::new((Name::l(name), ty)));
        return self;
    }

    pub fn add_test_def(mut self, name: &str) -> Self {
        self.defs.defs.push(Box::new((Name::l(name), self.types.get_test_type_type())));
        return self;
    }

    pub fn add_func_def(self, name: &str) -> FunctionDefBuilder {
        FunctionDefBuilder {
            parent_defs: self,
            params: Vec::new(),
            function_name: Name::l(name)
        }
    }

    pub fn destruct(self) -> (TypeVec, Definitions) {
        (self.types, self.defs)
    }
}

pub struct Definitions {
    defs: Vec<Box<dyn SymbolDefinition>>
}

pub struct DefsIter<'a> {
    iter: std::slice::Iter<'a, Box<dyn SymbolDefinition>>,
}

impl<'a> Iterator for DefsIter<'a> {
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|param| &**param as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Definitions {
    type IntoIter = DefsIter<'a>;

    fn enumerate(self) -> Self::IntoIter {
        DefsIter { iter: self.defs.iter() }
    }
}

impl Definitions {

    pub fn get(&self, name: &str) -> &dyn SymbolDefinition {
        &**self.defs.iter().find(|d| d.get_name().name == name).unwrap()
    }
}

pub struct FunctionDefBuilder {
    function_name: Name,
    params: Vec<DynRef<RefCell<Type>>>,
    parent_defs: EnvironmentBuilder
}

impl FunctionDefBuilder {
    pub fn add_array_param(mut self, base: PrimitiveType, dimension_count: usize) -> Self {
        self.params.push(self.parent_defs.types.get_array_type(base, dimension_count, true));
        return self;
    }

    pub fn add_view_param(mut self, base: PrimitiveType, dimension_count: usize) -> Self {
        self.params.push(self.parent_defs.types.get_view_type(base, dimension_count, true));
        return self;
    }

    pub fn return_type(mut self, base: PrimitiveType, dimension_count: usize) -> EnvironmentBuilder {
        let return_type = Some(self.parent_defs.types.get_array_type(base, dimension_count, true));
        let ty = self.parent_defs.types.get_function_type(self.params, return_type);
        self.parent_defs.defs.defs.push(Box::new((self.function_name, ty)));
        return self.parent_defs;
    }

    pub fn return_void(mut self) -> EnvironmentBuilder {
        let ty = self.parent_defs.types.get_function_type(self.params, None);
        self.parent_defs.defs.defs.push(Box::new((self.function_name, ty)));
        return self.parent_defs;
    }
}