use super::super::language::prelude::*;
use super::super::util::dynamic::Dynamic;

pub trait SymbolDefinition: Any + Dynamic + std::fmt::Debug {
    fn get_name(&self) -> &Name;

    fn get_type(&self) -> TypePtr;
}

impl SymbolDefinition for Declaration {
    fn get_name(&self) -> &Name {

        &self.variable
    }

    fn get_type(&self,) -> TypePtr {
        self.variable_type
    }
}

impl SymbolDefinition for Label {
    fn get_name(&self) -> &Name {

        &self.label
    }

    fn get_type(&self) -> TypePtr {
        unimplemented!()
    }
}

impl SymbolDefinition for Function {
    fn get_name(&self) -> &Name {

        &self.identifier
    }

    fn get_type(&self) -> TypePtr {
        self.function_type
    }
}

#[cfg(test)]
impl SymbolDefinition for Name {
    fn get_name(&self) -> &Name {
        self
    }

    fn get_type(&self) -> TypePtr {
        unimplemented!()
    }
}

#[cfg(test)]
impl SymbolDefinition for (Name, DynRef<std::cell::RefCell<Type>>) {
    fn get_name(&self) -> &Name {
        &self.0
    }

    fn get_type(&self) -> TypePtr {
        self.1
    }
}
