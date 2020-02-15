use std::vec::Vec;

use super::AstNode;
use super::position::{ TextPosition, BEGIN };
use super::identifier::{ Identifier, Name };
use super::print::{ Printer, Printable };

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

impl std::fmt::Display for Type
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
    {
        match self {
            Type::TestType => write!(f, "test"),
            Type::Primitive(PrimitiveType::Int) => write!(f, "int"),
            Type::Array(PrimitiveType::Int, dims) => write!(f, "int[{}]", ",".repeat(*dims as usize)),
            Type::Function(params, result) => {
                f.write_str("fn(")?;
                for param in params {
                    param.fmt(f)?;
                    f.write_str(", ")?;
                }
                f.write_str(")")?;
                if let Some(result_type) = result {
                    f.write_str(": ")?;
                    result_type.fmt(f)?;
                }
                Ok(())
            }
        }
    }
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

pub trait Statement: AstNode + Iterable<Expression> + Printable { }

#[derive(Debug, Eq)]
pub struct If
{
    pub pos: TextPosition,
    pub condition: Expression,
    pub body: Block
}

#[derive(Debug, Eq)]
pub struct While
{
    pub pos: TextPosition,
    pub condition: Expression,
    pub body: Block
}

#[derive(Debug, Eq)]
pub struct Assignment
{
    pub pos: TextPosition,
    pub assignee: Expression,
    pub value: Expression
}

#[derive(Debug, Eq)]
pub struct Declaration
{
    pub pos: TextPosition,
    pub variable: Name,
    pub variable_type: Type,
    pub value: Option<Expression>
}

#[derive(Debug, Eq)]
pub struct Return
{
    pub pos: TextPosition,
    pub value: Option<Expression>
}

#[derive(Debug, PartialEq, Eq)]
pub enum Expression
{
    Call(Box<FunctionCall>),
    Variable(Variable),
    Literal(Literal)
}

#[derive(Debug, Eq)]
pub struct FunctionCall
{
    pub pos: TextPosition,
    pub function: Expression,
    pub parameters: Vec<Expression>
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
        self.condition == rhs.condition && self.body == rhs.body
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
        self.condition == rhs.condition && self.body == rhs.body
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
        self.assignee == rhs.assignee && self.value == rhs.value
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

impl Statement for Expression { }

impl Statement for If { }

impl Statement for While { }

impl Statement for Return { }

impl Statement for Block { }

impl Statement for Declaration { }

impl Statement for Assignment { }

impl AstNode for Expression
{
    fn pos(&self) -> &TextPosition
    {
        match self {
            Expression::Call(call) => call.pos(),
            Expression::Variable(var) => var.pos(),
            Expression::Literal(lit) => lit.pos()
        }
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
        self.function == rhs.function && self.parameters == rhs.parameters
    }
}

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

impl Printable for Function
{
    fn print<'a>(&self, printer: &mut (dyn Printer + 'a))
    {
        printer.print_function_header(self);
        if let Some(ref body) = self.body {
            body.print(printer);
        }
    }
}

impl Printable for Block
{
    fn print<'a>(&self, printer: &mut (dyn Printer + 'a))
    {
        printer.enter_block();
        for statement in self.statements.iter() {
            statement.print(printer);
        }
        printer.exit_block();
    }
}

impl Printable for Expression
{
    fn print<'a>(&self, printer: &mut (dyn Printer + 'a))
    {
        printer.print_expression(self);
    }
}

impl Printable for If
{
    fn print<'a>(&self, printer: &mut (dyn Printer + 'a))
    {
        printer.print_if_header(self);
        self.body.print(printer);
    }
}

impl Printable for While
{
    fn print<'a>(&self, printer: &mut (dyn Printer + 'a))
    {
        printer.print_while_header(self);
        self.body.print(printer);
    }
}

impl Printable for Declaration
{
    fn print<'a>(&self, printer: &mut (dyn Printer + 'a))
    {
        printer.print_declaration(self);
    }
}

impl Printable for Return
{
    fn print<'a>(&self, printer: &mut (dyn Printer + 'a))
    {
        printer.print_return(self);
    }
}

impl Printable for Assignment
{
    fn print<'a>(&self, printer: &mut (dyn Printer + 'a))
    {
        printer.print_assignment(self);
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Expression
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)>
    {
        Box::new(std::iter::once(self))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)>
    {
        Box::new(std::iter::once(self))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for If
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)>
    {
        Box::new(std::iter::once(&self.condition))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)>
    {
        Box::new(std::iter::once(&mut self.condition))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for While
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)>
    {
        Box::new(std::iter::once(&self.condition))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)>
    {
        Box::new(std::iter::once(&mut self.condition))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Return
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)>
    {
        if let Some(ref val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)>
    {
        if let Some(ref mut val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Block
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)>
    {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)>
    {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Declaration
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)>
    {
        if let Some(ref val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)>
    {
        if let Some(ref mut val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Assignment
{
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)>
    {
        Box::new(std::iter::once(&self.assignee).chain(std::iter::once(&self.value)))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)>
    {
        Box::new(std::iter::once(&mut self.assignee).chain(std::iter::once(&mut self.value)))
    }
}
