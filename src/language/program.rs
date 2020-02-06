use std::vec::Vec;
use std::any::Any;

use super::super::util::dynamic::{ DynEq, Dynamic };

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identifier {
    name: String,
    id: u32
}

impl Identifier
{
	pub fn new(name: &str) -> Identifier
	{
        Identifier {
            name: name.to_owned(),
            id: 0
        }
	}

	pub fn auto(id: u32) -> Identifier
	{
		Identifier {
            name: "auto".to_owned(),
            id: id
        }
	}
}

impl PartialOrd for Identifier
{
	fn partial_cmp(&self, rhs: &Identifier) -> Option<std::cmp::Ordering>
	{
		Some(self.cmp(rhs))
	}
}

impl Ord for Identifier
{
	fn cmp(&self, rhs: &Identifier) -> std::cmp::Ordering
	{
        match self.name.cmp(&rhs.name) {
            std::cmp::Ordering::Equal => self.id.cmp(&rhs.id),
            x => x
        }
	}
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Literal {
	pub value: i32
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum PrimitiveType 
{
    Int
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type 
{
    Primitive(PrimitiveType),
    Array(PrimitiveType, u32),
    Function(Vec<Type>, Option<Box<Type>>)
}

#[derive(Debug, PartialEq, Eq)]
pub struct Function
{
    identifier: Identifier,
    params: Vec<(Identifier, Type)>,
    return_type: Type,
    body: Option<Block>
}

#[derive(Debug, PartialEq, Eq)]
pub struct Block
{
    statements: Vec<Box<dyn Statement>>
}

pub trait Statement: std::fmt::Debug + Any + DynEq + Dynamic
{

}

impl PartialEq for dyn Statement
{
    fn eq(&self, rhs: &dyn Statement) -> bool
    {
        self.dyn_eq(rhs.dynamic())
    }
}

impl Eq for dyn Statement {}

impl Statement for Block
{

}

pub trait Expression: std::fmt::Debug + Any + DynEq + Dynamic
{

}

impl PartialEq for dyn Expression
{
    fn eq(&self, rhs: &dyn Expression) -> bool
    {
        self.dyn_eq(rhs.dynamic())
    }
}

impl Eq for dyn Expression {}

impl Statement for dyn Expression 
{

}

#[derive(Debug, Eq)]
pub struct If
{
    condition: Box<dyn Expression>,
    body: Block
}

impl PartialEq for If
{
    fn eq(&self, rhs: &If) -> bool
    {
        *self.condition == *rhs.condition && self.body == rhs.body
    }
}

#[derive(Debug, Eq)]
pub struct While
{
    condition: Box<dyn Expression>,
    body: Block
}

impl PartialEq for While
{
    fn eq(&self, rhs: &While) -> bool
    {
        *self.condition == *rhs.condition && self.body == rhs.body
    }
}

#[derive(Debug, Eq)]
pub struct Assignment
{
    assignee: Box<dyn Expression>,
    value: Box<dyn Expression>
}

impl PartialEq for Assignment
{
    fn eq(&self, rhs: &Assignment) -> bool
    {
        *self.assignee == *rhs.assignee && *self.value == *rhs.value
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Declaration
{
    variable: Identifier,
    value: Option<Box<dyn Expression>>
}

#[derive(Debug, PartialEq, Eq)]
pub struct Return
{
    value: Option<Box<dyn Expression>>
}

#[derive(Debug, Eq)]
pub struct FunctionCall
{
    function: Box<dyn Expression>,
    parameters: Vec<Box<dyn Expression>>
}

impl PartialEq for FunctionCall
{
    fn eq(&self, rhs: &FunctionCall) -> bool
    {
        *self.function == *rhs.function && self.parameters == rhs.parameters
    }
}

impl Expression for FunctionCall
{

}

impl Expression for Identifier
{

}

impl Expression for Literal
{

}