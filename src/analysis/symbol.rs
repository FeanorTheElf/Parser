use super::super::language::prelude::*;

use super::super::util::dynamic::Dynamic;
use std::any::Any;

pub trait SymbolDefinition: Any + Dynamic + std::fmt::Debug {
    fn get_name(&self) -> &Name;

    fn calc_type(&self) -> Type;
}

impl SymbolDefinition for Declaration {
    fn get_name(&self) -> &Name {

        &self.variable
    }

    fn calc_type(&self) -> Type {

        self.variable_type.clone()
    }
}

impl SymbolDefinition for Label {
    fn get_name(&self) -> &Name {

        &self.label
    }

    fn calc_type(&self) -> Type {

        Type::JumpLabel
    }
}

impl SymbolDefinition for Function {
    fn get_name(&self) -> &Name {

        &self.identifier
    }

    fn calc_type(&self) -> Type {

        Type::Function(
            self.params
                .iter()
                .map(|p| Box::new(p.variable_type.clone()))
                .collect(),
            self.return_type.clone().map(Box::new),
        )
    }
}

#[cfg(test)]

impl SymbolDefinition for Name {
    fn get_name(&self) -> &Name {

        self
    }

    fn calc_type(&self) -> Type {

        Type::TestType
    }
}

#[cfg(test)]

impl SymbolDefinition for (Name, Type) {
    fn get_name(&self) -> &Name {

        &self.0
    }

    fn calc_type(&self) -> Type {

        self.1.clone()
    }
}
