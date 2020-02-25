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

impl SymbolDefinition for Label
{
    fn get_name(&self) -> &Name
    {
        &self.label
    }

    fn calc_type(&self) -> Type
    {
        Type::JumpLabel
    }
}

impl SymbolDefinition for (TextPosition, Name, Type)
{
    fn get_name(&self) -> &Name
    {
        &self.1
    }

    fn calc_type(&self) -> Type
    {
        self.2.clone()
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
        Type::Function(self.params.iter().map(|p| Box::new(p.2.clone())).collect(), self.return_type.clone().map(Box::new))
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