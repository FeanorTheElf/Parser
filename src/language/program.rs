use super::backend::{Backend, OutputError, Printable};
use super::error::{CompileError, ErrorType};
use super::identifier::{BuiltInIdentifier, Identifier, Name};
use super::position::{TextPosition, BEGIN};
use super::AstNode;
use super::statements::*;
use super::program_parallel_for::*;

use super::super::util::iterable::{Iterable, LifetimeIterable};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum PrimitiveType {
    Int,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    TestType,
    JumpLabel,
    Primitive(PrimitiveType),
    Array(PrimitiveType, u32),
    Function(Vec<Box<Type>>, Option<Box<Type>>),
    View(Box<Type>),
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Type::TestType => write!(f, "test"),
            Type::Primitive(PrimitiveType::Int) => write!(f, "int"),
            Type::Array(PrimitiveType::Int, dims) => {
                write!(f, "int[{}]", ",".repeat(*dims as usize))
            }
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
            Type::JumpLabel => write!(f, "LABEL"),
            Type::View(viewn_type) => write!(f, "&{}", viewn_type),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Program {
    pub items: Vec<Box<Function>>,
}

#[derive(Debug, Eq, Clone)]
pub struct Declaration {
    pub pos: TextPosition,
    pub variable: Name,
    pub variable_type: Type,
}

#[derive(Debug, Eq, Clone)]
pub struct Function {
    pub pos: TextPosition,
    pub identifier: Name,
    pub params: Vec<Declaration>,
    pub return_type: Option<Type>,
    pub body: Option<Block>,
}

#[derive(Debug, Eq, Clone)]
pub struct Block {
    pub pos: TextPosition,
    pub statements: Vec<Box<dyn Statement>>,
}

pub type TransformerResult<T: ?Sized> = Result<Box<T>, (Box<T>, CompileError)>;

pub trait StatementTransformer {
    fn transform_block(&mut self, statement: Box<Block>) -> TransformerResult<dyn Statement> ;
    fn transform_parallel_for(&mut self, statement: Box<ParallelFor>) -> TransformerResult<dyn Statement>;
    fn transform_label(&mut self, statement: Box<Label>) -> TransformerResult<dyn Statement>;
    fn transform_goto(&mut self, statement: Box<Goto>) -> TransformerResult<dyn Statement>;
    fn transform_if(&mut self, statement: Box<If>) -> TransformerResult<dyn Statement>;
    fn transform_while(&mut self, statement: Box<While>) -> TransformerResult<dyn Statement>;
    fn transform_return(&mut self, statement: Box<Return>) -> TransformerResult<dyn Statement>;
    fn transform_assignment(&mut self, statement: Box<Assignment>) -> TransformerResult<dyn Statement>;
    fn transform_declaration(&mut self, statement: Box<LocalVariableDeclaration>) -> TransformerResult<dyn Statement>;
    fn transform_expression(&mut self, statement: Box<Expression>) -> TransformerResult<dyn Statement>;
}

pub trait Statement: AstNode + Iterable<Expression> + Iterable<Block> + Printable {
    fn dyn_clone(&self) -> Box<dyn Statement>;
    fn transform(self: Box<Self>, transformer: &mut dyn StatementTransformer) -> TransformerResult<dyn Statement>;
    fn transform_children(&mut self, transformer: &mut dyn StatementTransformer) -> Result<(), CompileError>;
}

impl Clone for Box<dyn Statement> {
    fn clone(&self) -> Box<dyn Statement> {
        self.dyn_clone()
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expression {
    Call(Box<FunctionCall>),
    Variable(Variable),
    Literal(Literal),
}

#[derive(Debug, Eq, Clone)]
pub struct FunctionCall {
    pub pos: TextPosition,
    pub function: Expression,
    pub parameters: Vec<Expression>,
}

#[derive(Debug, Eq, Clone)]
pub struct Variable {
    pub pos: TextPosition,
    pub identifier: Identifier,
}

#[derive(Debug, Eq, Clone)]
pub struct Literal {
    pub pos: TextPosition,
    pub value: i32,
}

impl Expression {
    pub fn expect_identifier(&self) -> Result<&Identifier, CompileError> {
        match self {
            Expression::Call(_) => Err(CompileError::new(
                self.pos(),
                format!("A function call is not allowed here."),
                ErrorType::VariableRequired,
            )),
            Expression::Variable(var) => Ok(&var.identifier),
            Expression::Literal(_) => Err(CompileError::new(
                self.pos(),
                format!("A function call is not allowed here."),
                ErrorType::VariableRequired,
            )),
        }
    }
}

impl AstNode for Program {
    fn pos(&self) -> &TextPosition {
        &BEGIN
    }
}

impl AstNode for Declaration {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for Declaration {
    fn eq(&self, rhs: &Declaration) -> bool {
        self.variable == rhs.variable && self.variable_type == rhs.variable_type
    }
}

impl AstNode for Function {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for Function {
    fn eq(&self, rhs: &Function) -> bool {
        self.identifier == rhs.identifier
            && self.params == rhs.params
            && self.return_type == rhs.return_type
            && self.body == rhs.body
    }
}

impl AstNode for Block {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for Block {
    fn eq(&self, rhs: &Block) -> bool {
        self.statements == rhs.statements
    }
}

impl PartialEq for dyn Statement {
    fn eq(&self, rhs: &dyn Statement) -> bool {
        self.dyn_eq(rhs.dynamic())
    }
}

impl Eq for dyn Statement {}

impl Statement for Expression {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn transform(self: Box<Self>, transformer: &mut dyn StatementTransformer) -> TransformerResult<dyn Statement> {
        transformer.transform_expression(self)
    }

    fn transform_children(&mut self, _transformer: &mut dyn StatementTransformer) -> Result<(), CompileError> {
        Ok(())
    }
}

impl AstNode for Expression {
    fn pos(&self) -> &TextPosition {
        match self {
            Expression::Call(call) => call.pos(),
            Expression::Variable(var) => var.pos(),
            Expression::Literal(lit) => lit.pos(),
        }
    }
}

impl PartialEq<BuiltInIdentifier> for Expression {
    fn eq(&self, rhs: &BuiltInIdentifier) -> bool {
        match self {
            Expression::Variable(var) => var.identifier == Identifier::BuiltIn(*rhs),
            _ => false,
        }
    }
}

impl AstNode for FunctionCall {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for FunctionCall {
    fn eq(&self, rhs: &FunctionCall) -> bool {
        self.function == rhs.function && self.parameters == rhs.parameters
    }
}

impl PartialEq for Variable {
    fn eq(&self, rhs: &Variable) -> bool {
        self.identifier == rhs.identifier
    }
}

impl AstNode for Variable {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for Literal {
    fn eq(&self, rhs: &Literal) -> bool {
        self.value == rhs.value
    }
}

impl AstNode for Literal {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl Printable for Program {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        for item in &self.items {
            item.print(printer)?;
        }
        Ok(())
    }
}

impl Printable for Function {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.print_function_header(self)?;
        if let Some(ref body) = self.body {
            body.print(printer)?;
        }
        Ok(())
    }
}

impl Printable for Block {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.enter_block()?;
        for statement in self.statements.iter() {
            statement.print(printer)?;
        }
        printer.exit_block()?;
        Ok(())
    }
}

impl Printable for Expression {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.print_expression(self)
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Expression {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(self))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(self))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Block {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for Expression {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for Block {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(self))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(self))
    }
}
