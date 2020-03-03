use super::super::language::prelude::*;

pub trait SymbolDefinition
{
    fn get_name(&self) -> &Name;
    fn calc_type(&self) -> Type;
}

impl SymbolDefinition for Declaration
{
    fn get_name(&self) -> &Name
    {
        &self.variable
    }

    fn calc_type(&self) -> Type
    {
        self.variable_type.clone()
    }
}

impl SymbolDefinition for FormalParameter
{
    fn get_name(&self) -> &Name
    {
        &self.name
    }

    fn calc_type(&self) -> Type
    {
        self.param_type.clone()
    }
}

impl SymbolDefinition for Function
{
    fn get_name(&self) -> &Name
    {
        &self.identifier
    }

    fn calc_type(&self) -> Type
    {
        Type::Function(self.params.iter().map(|p| Box::new(p.param_type.clone())).collect(), self.return_type.clone().map(Box::new))
    }
}

#[cfg(test)]
impl SymbolDefinition for Name
{
    fn get_name(&self) -> &Name
    {
        self
    }

    fn calc_type(&self) -> Type
    {
        Type::TestType
    }
}