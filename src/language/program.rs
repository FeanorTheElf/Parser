use super::error::{CompileError, ErrorType};
use super::identifier::{BuiltInIdentifier, Identifier, Name};
use super::position::{TextPosition, BEGIN};
use super::types::{Type, FunctionType, TypePtr, TypeVec};
use super::AstNode;

use super::super::util::iterable::{Iterable, LifetimeIterable};
use super::super::util::cmp::Comparing;
use super::super::util::dyn_lifetime::*;

use std::cell::Ref;

#[derive(Debug)]
pub struct Program {
    pub items: Vec<Box<Function>>,
    pub types: TypeVec
}

impl Program {
    pub fn work<'a>(&'a mut self) -> (&'a mut Vec<Box<Function>>, Lifetime<'a>) {
        (&mut self.items, self.types.get_lifetime())
    }
}

#[derive(Debug, Eq, Clone)]
pub struct Declaration {
    pub pos: TextPosition,
    pub variable: Name,
    pub variable_type: TypePtr,
}

#[derive(Debug, Eq, Clone)]
pub struct Function {
    pub pos: TextPosition,
    pub identifier: Name,
    pub params: Vec<Declaration>,
    pub function_type: TypePtr,
    pub body: Option<Block>,
}

impl Function {
    pub fn get_type<'a, 'b: 'a>(&'a self, prog_lifetime: Lifetime<'b>) -> Ref<'a, FunctionType> {
        Ref::map(prog_lifetime.cast(self.function_type).borrow(), |f| match f {
            Type::Function(func) => func,
            ty => panic!("Function definition has type {}", ty)
        })
    }
}

#[derive(Debug, Eq, Clone)]
pub struct Block {
    pub pos: TextPosition,
    pub statements: Vec<Box<dyn Statement>>,
}

pub trait Statement: AstNode + Iterable<Expression> + Iterable<Block> {
    fn dyn_clone(&self) -> Box<dyn Statement>;
}

impl Clone for Box<dyn Statement> {
    fn clone(&self) -> Box<dyn Statement> {

        self.dyn_clone()
    }
}

#[derive(Debug, Eq, Clone)]

pub struct If {
    pub pos: TextPosition,
    pub condition: Expression,
    pub body: Block,
}

#[derive(Debug, Eq, Clone)]

pub struct While {
    pub pos: TextPosition,
    pub condition: Expression,
    pub body: Block,
}

#[derive(Debug, Eq, Clone)]

pub struct Assignment {
    pub pos: TextPosition,
    pub assignee: Expression,
    pub value: Expression,
}

#[derive(Debug, Eq, Clone)]

pub struct LocalVariableDeclaration {
    pub declaration: Declaration,
    pub value: Option<Expression>,
}

#[derive(Debug, Eq, Clone)]

pub struct Return {
    pub pos: TextPosition,
    pub value: Option<Expression>,
}

#[derive(Debug, Eq, Clone)]

pub struct Label {
    pub pos: TextPosition,
    pub label: Name,
}

#[derive(Debug, Eq, Clone)]

pub struct Goto {
    pub pos: TextPosition,
    pub target: Name,
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

impl Block {
    pub fn scan_top_level_expressions<'a, F>(&'a self, f: &mut F)
    where
        F: FnMut(&'a Expression),
    {

        for statement in &self.statements {

            for expr in statement.iter() {

                f(expr);
            }

            for sub_block in statement.iter() {

                (sub_block as &'a Block).scan_top_level_expressions(f);
            }
        }
    }
}

impl Expression {
    pub fn expect_identifier(&self) -> Result<&Variable, CompileError> {

        match self {
            Expression::Call(_) => Err(CompileError::new(
                self.pos(),
                format!("Only a variable is allowed here."),
                ErrorType::VariableRequired,
            )),
            Expression::Variable(var) => Ok(&var),
            Expression::Literal(_) => Err(CompileError::new(
                self.pos(),
                format!("Only a variable is allowed here."),
                ErrorType::VariableRequired,
            )),
        }
    }

    pub fn is_lvalue(&self) -> bool {

        match self {
            Expression::Call(call) => match &call.function {
                Expression::Call(_) => false,
                Expression::Literal(_) => {

                    debug_assert!(false);

                    false
                }
                Expression::Variable(var) => {
                    var.identifier == Identifier::BuiltIn(BuiltInIdentifier::FunctionIndex)
                        && call.parameters[0].is_lvalue()
                }
            },
            Expression::Variable(_) => true,
            Expression::Literal(_) => false,
        }
    }
}

impl PartialEq<Identifier> for Expression {
    fn eq(&self, rhs: &Identifier) -> bool {

        match self {
            Expression::Variable(var) => var.identifier == *rhs,
            _ => false,
        }
    }
}

impl AstNode for Program {
    fn pos(&self) -> &TextPosition {

        &BEGIN
    }
}

impl PartialEq for Program {
    fn eq(&self, rhs: &Program) -> bool {
        if self.items.len() != rhs.items.len() {
            return false;
        }
        let cmp_fn = |lhs: &&Function, rhs: &&Function| lhs.identifier.cmp(&rhs.identifier);
        let mut self_items = self.items.iter().map(|f| Comparing::new(&**f, cmp_fn)).collect::<Vec<_>>();
        let mut rhs_items = self.items.iter().map(|f| Comparing::new(&**f, cmp_fn)).collect::<Vec<_>>();
        self_items.sort();
        rhs_items.sort();
        for i in 0..self_items.len() {
            if self_items[i] != rhs_items[i] {
                return false;
            }
        }
        return true;
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
            && self.function_type == rhs.function_type
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

impl AstNode for If {
    fn pos(&self) -> &TextPosition {

        &self.pos
    }
}

impl PartialEq for If {
    fn eq(&self, rhs: &If) -> bool {

        self.condition == rhs.condition && self.body == rhs.body
    }
}

impl AstNode for While {
    fn pos(&self) -> &TextPosition {

        &self.pos
    }
}

impl PartialEq for While {
    fn eq(&self, rhs: &While) -> bool {

        self.condition == rhs.condition && self.body == rhs.body
    }
}

impl AstNode for Assignment {
    fn pos(&self) -> &TextPosition {

        &self.pos
    }
}

impl PartialEq for Assignment {
    fn eq(&self, rhs: &Assignment) -> bool {

        self.assignee == rhs.assignee && self.value == rhs.value
    }
}

impl AstNode for LocalVariableDeclaration {
    fn pos(&self) -> &TextPosition {

        self.declaration.pos()
    }
}

impl PartialEq for LocalVariableDeclaration {
    fn eq(&self, rhs: &LocalVariableDeclaration) -> bool {

        self.declaration == rhs.declaration && self.value == rhs.value
    }
}

impl AstNode for Return {
    fn pos(&self) -> &TextPosition {

        &self.pos
    }
}

impl PartialEq for Label {
    fn eq(&self, rhs: &Label) -> bool {

        self.label == rhs.label
    }
}

impl AstNode for Label {
    fn pos(&self) -> &TextPosition {

        &self.pos
    }
}

impl PartialEq for Return {
    fn eq(&self, rhs: &Return) -> bool {

        self.value == rhs.value
    }
}

impl PartialEq for Goto {
    fn eq(&self, rhs: &Goto) -> bool {

        self.target == rhs.target
    }
}

impl AstNode for Goto {
    fn pos(&self) -> &TextPosition {

        &self.pos
    }
}

impl Statement for Expression {
    fn dyn_clone(&self) -> Box<dyn Statement> {

        Box::new(self.clone())
    }
}

impl Statement for If {
    fn dyn_clone(&self) -> Box<dyn Statement> {

        Box::new(self.clone())
    }
}

impl Statement for While {
    fn dyn_clone(&self) -> Box<dyn Statement> {

        Box::new(self.clone())
    }
}

impl Statement for Return {
    fn dyn_clone(&self) -> Box<dyn Statement> {

        Box::new(self.clone())
    }
}

impl Statement for Block {
    fn dyn_clone(&self) -> Box<dyn Statement> {

        Box::new(self.clone())
    }
}

impl Statement for LocalVariableDeclaration {
    fn dyn_clone(&self) -> Box<dyn Statement> {

        Box::new(self.clone())
    }
}

impl Statement for Assignment {
    fn dyn_clone(&self) -> Box<dyn Statement> {

        Box::new(self.clone())
    }
}

impl Statement for Label {
    fn dyn_clone(&self) -> Box<dyn Statement> {

        Box::new(self.clone())
    }
}

impl Statement for Goto {
    fn dyn_clone(&self) -> Box<dyn Statement> {

        Box::new(self.clone())
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

impl<'a> LifetimeIterable<'a, Expression> for Expression {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {

        Box::new(std::iter::once(self))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {

        Box::new(std::iter::once(self))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for If {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {

        Box::new(std::iter::once(&self.condition))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {

        Box::new(std::iter::once(&mut self.condition))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for While {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {

        Box::new(std::iter::once(&self.condition))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {

        Box::new(std::iter::once(&mut self.condition))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Return {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {

        if let Some(ref val) = self.value {

            Box::new(std::iter::once(val))
        } else {

            Box::new(std::iter::empty())
        }
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {

        if let Some(ref mut val) = self.value {

            Box::new(std::iter::once(val))
        } else {

            Box::new(std::iter::empty())
        }
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

impl<'a> LifetimeIterable<'a, Expression> for LocalVariableDeclaration {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {

        if let Some(ref val) = self.value {

            Box::new(std::iter::once(val))
        } else {

            Box::new(std::iter::empty())
        }
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {

        if let Some(ref mut val) = self.value {

            Box::new(std::iter::once(val))
        } else {

            Box::new(std::iter::empty())
        }
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Assignment {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {

        Box::new(std::iter::once(&self.assignee).chain(std::iter::once(&self.value)))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {

        Box::new(std::iter::once(&mut self.assignee).chain(std::iter::once(&mut self.value)))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Label {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {

        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {

        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Goto {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {

        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {

        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for LocalVariableDeclaration {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {

        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {

        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for Assignment {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {

        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {

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

impl<'a> LifetimeIterable<'a, Block> for Return {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {

        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {

        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for Goto {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {

        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {

        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for Label {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {

        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {

        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for If {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {

        Box::new(std::iter::once(&self.body))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {

        Box::new(std::iter::once(&mut self.body))
    }
}

impl<'a> LifetimeIterable<'a, Block> for While {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {

        Box::new(std::iter::once(&self.body))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {

        Box::new(std::iter::once(&mut self.body))
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
