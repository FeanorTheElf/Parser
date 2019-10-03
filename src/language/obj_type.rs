use super::super::parser::prelude::*;
use super::scope::{ SymbolDefinition, SymbolDefinitionKind };

#[derive(Debug, PartialEq, Eq)]
pub enum PrimitiveType {
    Int
}

#[derive(Debug, PartialEq, Eq)]
pub enum Type {
    Primitive(PrimitiveType),
    Array(PrimitiveType, u32),
    Function(Vec<Type>, Option<Box<Type>>)
}

impl Type {
    pub fn from(t: &dyn TypeNode) -> Option<Type> {
        match t.get_kind() {
            TypeKind::Array(ref arr) => {
                if arr.get_dims() == 0 {
                    Some(Type::Primitive(PrimitiveType::Int))
                } else {
                    Some(Type::Array(PrimitiveType::Int, arr.get_dims()))
                }
            },
            TypeKind::Void(ref void) => {
                None
            }
        }
    }
    
    pub fn calc_from(definition: &dyn SymbolDefinition) -> Result<Type, CompileError> {
        match definition.get_kind() {
            SymbolDefinitionKind::Function(ref function) => {
                let param_types = Result::from(function.params.iter().map(|param| Type::calc_from(&**param)).collect())?;
                return Ok(Type::Function(param_types, Type::from(&*function.result).map(|t| Box::new(t))));
            },
            SymbolDefinitionKind::LocalVar(ref var) => {
                if let Some(var_type) = Type::from(&*var.variable_type) {
                    return Ok(var_type);
                } else {
                    return Err(CompileError::new(definition.get_annotation().clone(),
                        format!("Local variable cannot have type void")));
                }
            },
            SymbolDefinitionKind::Parameter(ref param) => {
                if let Some(var_type) = Type::from(&*param.param_type) {
                    return Ok(var_type);
                } else {
                    return Err(CompileError::new(definition.get_annotation().clone(),
                        format!("Parameter cannot have type void")));
                }
            }
        }
    }
}
