use super::super::lexer::error::CompileError;

#[derive(Debug, PartialEq, Eq)]
pub enum PrimitiveType 
{
    Int
}

#[derive(Debug, PartialEq, Eq)]
pub enum Type 
{
    Primitive(PrimitiveType),
    Array(PrimitiveType, u32),
    Function(Vec<Type>, Option<Box<Type>>)
}

pub trait TypeDefinition 
{
    fn calc_type(&self) -> Result<Option<Type>, CompileError>;
}
