use super::super::parser::prelude::*;
use super::super::parser::obj_type::*;

pub trait SymbolDefinition: Node 
{
    fn get_ident(&self) -> &Identifier;
    fn calc_type(&self) -> Result<Type, CompileError>;
}

impl SymbolDefinition for ParameterNode 
{
    fn get_ident(&self) -> &Identifier 
    {
        &self.ident
    }

    fn calc_type(&self) -> Result<Type, CompileError> 
    {
        if let Some(var_type) = self.param_type.calc_type()? {
            return Ok(var_type);
        } else {
            return Err(CompileError::new(self.get_annotation().clone(),
                format!("Parameter cannot have type void"), ErrorType::VariableVoidType));
        }
    }
}

impl SymbolDefinition for VariableDeclarationNode 
{
    fn get_ident(&self) -> &Identifier 
    {
        &self.ident
    }

    fn calc_type(&self) -> Result<Type, CompileError>
    {
        if let Some(var_type) = self.variable_type.calc_type()? {
            return Ok(var_type);
        } else {
            return Err(CompileError::new(self.get_annotation().clone(),
                format!("Local variable cannot have type void"), ErrorType::VariableVoidType));
        }
    }
}

impl SymbolDefinition for FunctionNode 
{
    fn get_ident(&self) -> &Identifier 
    {
        &self.ident
    }

    fn calc_type(&self) -> Result<Type, CompileError> 
    {
        let param_types = Result::from(self.params.iter().map(|param| param.calc_type()).collect())?;
        return Ok(Type::Function(param_types, self.result.calc_type()?.map(|t| Box::new(t))));
    }
}
