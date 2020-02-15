use std::vec::Vec;

use super::AstNode;
use super::position::{ TextPosition, BEGIN };
use super::identifier::{ Identifier, Name };

use super::super::util::iterable::{ Iterable, LifetimeIterable };

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum PrimitiveType 
{
    Int
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type
{
    TestType,
    Primitive(PrimitiveType),
    Array(PrimitiveType, u32),
    Function(Vec<Box<Type>>, Option<Box<Type>>)
}

#[derive(Debug, PartialEq, Eq)]
pub struct Program
{
    pub items: Vec<Box<Function>>
}

#[derive(Debug, Eq)]
pub struct Function
{
    pub pos: TextPosition,
    pub identifier: Name,
    pub params: Vec<(TextPosition, Name, Type)>,
    pub return_type: Option<Type>,
    pub body: Option<Block>
}

#[derive(Debug, Eq)]
pub struct Block
{
    pub pos: TextPosition,
    pub statements: Vec<Box<dyn Statement>>
}

pub trait Statement: AstNode + Iterable<Box<dyn Expression>> { }

#[derive(Debug, Eq)]
pub struct If
{
    pub pos: TextPosition,
    pub condition: Box<dyn Expression>,
    pub body: Block
}

#[derive(Debug, Eq)]
pub struct While
{
    pub pos: TextPosition,
    pub condition: Box<dyn Expression>,
    pub body: Block
}

#[derive(Debug, Eq)]
pub struct Assignment
{
    pub pos: TextPosition,
    pub assignee: Box<dyn Expression>,
    pub value: Box<dyn Expression>
}

#[derive(Debug, Eq)]
pub struct Declaration
{
    pub pos: TextPosition,
    pub variable: Name,
    pub variable_type: Type,
    pub value: Option<Box<dyn Expression>>
}

#[derive(Debug, Eq)]
pub struct Return
{
    pub pos: TextPosition,
    pub value: Option<Box<dyn Expression>>
}

pub trait Expression: AstNode { }

#[derive(Debug, Eq)]
pub struct FunctionCall
{
    pub pos: TextPosition,
    pub function: Box<dyn Expression>,
    pub parameters: Vec<Box<dyn Expression>>
}

#[derive(Debug, Eq, Clone)]
pub struct Variable {
    pub pos: TextPosition,
	pub identifier: Identifier
}

#[derive(Debug, Eq, Clone)]
pub struct Literal {
    pub pos: TextPosition,
	pub value: i32
}

impl AstNode for Program
{
    fn pos(&self) -> &TextPosition 
    {
        &BEGIN
    }
}

impl AstNode for Function
{
    fn pos(&self) -> &TextPosition
    {
        &self.pos
    }
}

impl PartialEq for Function
{
    fn eq(&self, rhs: &Function) -> bool
    {
        self.identifier == rhs.identifier && self.params == rhs.params && self.return_type == rhs.return_type && self.body == rhs.body
    }
}

impl AstNode for Block
{
    fn pos(&self) -> &TextPosition
    {
        &self.pos
    }
}

impl PartialEq for Block
{
    fn eq(&self, rhs: &Block) -> bool
    {
        self.statements == rhs.statements
    }
}

impl PartialEq for dyn Statement
{
    fn eq(&self, rhs: &dyn Statement) -> bool
    {
        self.dyn_eq(rhs.dynamic())
    }
}

impl Eq for dyn Statement {}

impl AstNode for If
{
    fn pos(&self) -> &TextPosition
    {
        &self.pos
    }
}

impl PartialEq for If
{
    fn eq(&self, rhs: &If) -> bool
    {
        *self.condition == *rhs.condition && self.body == rhs.body
    }
}

impl AstNode for While
{
    fn pos(&self) -> &TextPosition
    {
        &self.pos
    }
}

impl PartialEq for While
{
    fn eq(&self, rhs: &While) -> bool
    {
        *self.condition == *rhs.condition && self.body == rhs.body
    }
}

impl AstNode for Assignment
{
    fn pos(&self) -> &TextPosition
    {
        &self.pos
    }
}

impl PartialEq for Assignment
{
    fn eq(&self, rhs: &Assignment) -> bool
    {
        *self.assignee == *rhs.assignee && *self.value == *rhs.value
    }
}

impl AstNode for Declaration
{
    fn pos(&self) -> &TextPosition
    {
        &self.pos
    }
}

impl PartialEq for Declaration
{
    fn eq(&self, rhs: &Declaration) -> bool
    {
        self.variable == rhs.variable && self.variable_type == rhs.variable_type && self.value == rhs.value
    }
}

impl AstNode for Return
{
    fn pos(&self) -> &TextPosition
    {
        &self.pos
    }
}

impl PartialEq for Return
{
    fn eq(&self, rhs: &Return) -> bool
    {
        self.value == rhs.value
    }
}

impl Statement for Box<dyn Expression> { }

impl Statement for If { }

impl Statement for While { }

impl Statement for Return { }

impl Statement for Block { }

impl Statement for Declaration { }

impl Statement for Assignment { }

impl PartialEq for dyn Expression
{
    fn eq(&self, rhs: &dyn Expression) -> bool
    {
        self.dyn_eq(rhs.dynamic())
    }
}

impl Eq for dyn Expression {}

impl AstNode for Box<dyn Expression>
{
    fn pos(&self) -> &TextPosition
    {
        (**self).pos()
    }
}

impl AstNode for FunctionCall
{
    fn pos(&self) -> &TextPosition
    {
        &self.pos
    }
}

impl PartialEq for FunctionCall
{
    fn eq(&self, rhs: &FunctionCall) -> bool
    {
        *self.function == *rhs.function && self.parameters == rhs.parameters
    }
}

impl Expression for FunctionCall { }

impl PartialEq for Variable
{
    fn eq(&self, rhs: &Variable) -> bool
    {
        self.identifier == rhs.identifier
    }
}

impl AstNode for Variable
{
    fn pos(&self) -> &TextPosition
    {
        &self.pos
    }
}

impl Expression for Variable { }

impl PartialEq for Literal
{
    fn eq(&self, rhs: &Literal) -> bool
    {
        self.value == rhs.value
    }
}

impl AstNode for Literal
{
    fn pos(&self) -> &TextPosition
    {
        &self.pos
    }
}

impl Expression for Literal { }

impl<'a> LifetimeIterable<'a, Box<dyn Expression>> for Box<dyn Expression>
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Box<dyn Expression>> + 'a)>
    {
        Box::new(std::iter::once(self))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Box<dyn Expression>> + 'a)>
    {
        Box::new(std::iter::once(self))
    }
}

impl<'a> LifetimeIterable<'a, Box<dyn Expression>> for If
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Box<dyn Expression>> + 'a)>
    {
        Box::new(std::iter::once(&self.condition))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Box<dyn Expression>> + 'a)>
    {
        Box::new(std::iter::once(&mut self.condition))
    }
}

impl<'a> LifetimeIterable<'a, Box<dyn Expression>> for While
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Box<dyn Expression>> + 'a)>
    {
        Box::new(std::iter::once(&self.condition))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Box<dyn Expression>> + 'a)>
    {
        Box::new(std::iter::once(&mut self.condition))
    }
}

impl<'a> LifetimeIterable<'a, Box<dyn Expression>> for Return
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Box<dyn Expression>> + 'a)>
    {
        if let Some(ref val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Box<dyn Expression>> + 'a)>
    {
        if let Some(ref mut val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }
}

impl<'a> LifetimeIterable<'a, Box<dyn Expression>> for Block
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Box<dyn Expression>> + 'a)>
    {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Box<dyn Expression>> + 'a)>
    {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Box<dyn Expression>> for Declaration
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Box<dyn Expression>> + 'a)>
    {
        if let Some(ref val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Box<dyn Expression>> + 'a)>
    {
        if let Some(ref mut val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }
}

impl<'a> LifetimeIterable<'a, Box<dyn Expression>> for Assignment
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Box<dyn Expression>> + 'a)>
    {
        Box::new(std::iter::once(&self.assignee).chain(std::iter::once(&self.value)))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Box<dyn Expression>> + 'a)>
    {
        Box::new(std::iter::once(&mut self.assignee).chain(std::iter::once(&mut self.value)))
    }
}
