use super::super::language::prelude::*;
use super::super::util::dynamic::Dynamic;

use super::super::util::dyn_lifetime::*;
use std::any::Any;

pub trait SymbolDefinition: Any + Dynamic + std::fmt::Debug {
    fn get_name(&self) -> &Name;

    fn calc_type(&self, prog_lifetime: Lifetime) -> Type;
}

impl SymbolDefinition for Declaration {
    fn get_name(&self) -> &Name {

        &self.variable
    }

    fn calc_type(&self, prog_lifetime: Lifetime) -> Type {

        prog_lifetime.cast(self.variable_type).borrow().clone()
    }
}

impl SymbolDefinition for Label {
    fn get_name(&self) -> &Name {

        &self.label
    }

    fn calc_type(&self, _prog_lifetime: Lifetime) -> Type {

        Type::JumpLabel
    }
}

impl SymbolDefinition for Function {
    fn get_name(&self) -> &Name {

        &self.identifier
    }

    fn calc_type(&self, _prog_lifetime: Lifetime) -> Type {
        Type::Function(FunctionType {
            param_types: self.params
                .iter()
                .map(|p| p.variable_type)
                .collect(),
            return_type: self.return_type,
        })
    }
}

#[cfg(test)]
impl SymbolDefinition for Name {
    fn get_name(&self) -> &Name {
        self
    }

    fn calc_type(&self, _prog_lifetime: Lifetime) -> Type {
        Type::TestType
    }
}

#[cfg(test)]
impl SymbolDefinition for (Name, Type) {
    fn get_name(&self) -> &Name {
        &self.0
    }

    fn calc_type(&self, _prog_lifetime: Lifetime) -> Type {
        self.1.clone()
    }
}

#[cfg(test)]
impl SymbolDefinition for (Name, DynRef<std::cell::RefCell<Type>>) {
    fn get_name(&self) -> &Name {
        &self.0
    }

    fn calc_type(&self, prog_lifetime: Lifetime) -> Type {
        prog_lifetime.cast(self.1).borrow().clone()
    }
}
