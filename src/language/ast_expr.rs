use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::{Identifier, Name, BuiltInIdentifier};
use super::ast::*;
use super::types::*;

use std::cell::RefCell;

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
    pub result_type_cache: RefCell<Option<VoidableType>>
}

impl AstNode for FunctionCall {} 

impl AstNodeFuncs for FunctionCall {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for FunctionCall {
    fn eq(&self, rhs: &FunctionCall) -> bool {
        self.function == rhs.function && self.parameters == rhs.parameters
    }
}

#[derive(Debug, Eq, Clone)]
pub struct Variable {
    pub pos: TextPosition,
    pub identifier: Identifier,
}

impl PartialEq<Identifier> for Variable {
    fn eq(&self, rhs: &Identifier) -> bool {
        &self.identifier == rhs
    }
}

impl PartialEq<Name> for Variable {
    fn eq(&self, rhs: &Name) -> bool {
        &self.identifier == rhs
    }
}

impl PartialEq for Variable {
    fn eq(&self, rhs: &Variable) -> bool {
        self.identifier == rhs.identifier
    }
}

impl AstNode for Variable {} 

impl AstNodeFuncs for Variable {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

#[derive(Debug, Eq, Clone)]
pub struct Literal {
    pub pos: TextPosition,
    pub value: i32,
    pub literal_type: Type
}

impl PartialEq for Literal {
    fn eq(&self, rhs: &Literal) -> bool {
        self.value == rhs.value
    }
}

impl AstNode for Literal {} 

impl AstNodeFuncs for Literal {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl Expression {

    pub fn traverse_preorder<'a, F>(&'a self, f: &mut F) -> Result<(), CompileError>
    where F: FnMut(&'a Expression) -> TraversePreorderResult
    {
        let result = f(self);
        match result {
            Err(TraversePreorderCancel::RealError(e)) => return Err(e),
            Err(TraversePreorderCancel::DoNotRecurse) => {},
            Ok(()) => match self {
                Expression::Call(call) => {
                    call.function.traverse_preorder(f)?;
                    for p in &call.parameters {
                        p.traverse_preorder(f)?;
                    }
                },
                Expression::Variable(_) | Expression::Literal(_) => {}
            }
        };
        return Ok(());
    }

    pub fn traverse_preorder_mut<F>(&mut self, f: &mut F) -> Result<(), CompileError>
    where F: FnMut(&mut Expression) -> TraversePreorderResult
    {
        let result = f(self);
        match result {
            Err(TraversePreorderCancel::RealError(e)) => return Err(e),
            Err(TraversePreorderCancel::DoNotRecurse) => {},
            Ok(()) => match self {
                Expression::Call(call) => {
                    call.function.traverse_preorder_mut(f)?;
                    for p in &mut call.parameters {
                        p.traverse_preorder_mut(f)?;
                    }
                },
                Expression::Variable(_) | Expression::Literal(_) => {}
            }
        };
        return Ok(());
    }
}

pub struct ExpressionIdentifierIter<'a> {
    iters: Vec<&'a Expression>
}

impl<'a> Iterator for ExpressionIdentifierIter<'a> {
    type Item = &'a Identifier;

    fn next(&mut self) -> Option<&'a Identifier> {
        loop {
            match self.iters.pop() {
                None => {
                    return None;
                },
                Some(Expression::Call(call)) => {
                    self.iters.push(&call.function);
                    self.iters.extend(call.parameters.iter());
                },
                Some(Expression::Variable(var)) => return Some(&var.identifier),
                Some(Expression::Literal(_)) => {}
            }
        }
    }
}

pub struct ExpressionIdentifierIterMut<'a> {
    iters: Vec<&'a mut Expression>
}

impl<'a> Iterator for ExpressionIdentifierIterMut<'a> {
    type Item = &'a mut Identifier;

    fn next(&mut self) -> Option<&'a mut Identifier> {
        loop {
            match self.iters.pop() {
                None => {
                    return None;
                },
                Some(Expression::Call(call)) => {
                    self.iters.push(&mut call.function);
                    self.iters.extend(call.parameters.iter_mut());
                },
                Some(Expression::Variable(var)) => return Some(&mut var.identifier),
                Some(Expression::Literal(_)) => {}
            }
        }
    }
}

impl Expression {
    
    pub fn identifiers<'a>(&'a self) -> ExpressionIdentifierIter<'a> {
        ExpressionIdentifierIter {
            iters: vec![self]
        }
    }

    pub fn identifiers_mut<'a>(&'a mut self) -> ExpressionIdentifierIterMut<'a> {
        ExpressionIdentifierIterMut {
            iters: vec![self]
        }
    }

    pub fn names<'a>(&'a self) -> impl Iterator<Item = &'a Name> {
        self.identifiers().filter_map(Identifier::as_name)
    }

    pub fn names_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Name> {
        self.identifiers_mut().filter_map(Identifier::as_name_mut)
    }
}

impl From<FunctionCall> for Expression {
    fn from(call: FunctionCall) -> Expression {
        Expression::Call(Box::new(call))
    }
}

impl From<Variable> for Expression {
    fn from(var: Variable) -> Expression {
        Expression::Variable(var)
    }
}

impl From<Literal> for Expression {
    fn from(literal: Literal) -> Expression {
        Expression::Literal(literal)
    }
}

impl PartialEq<FunctionCall> for Expression {
    fn eq(&self, rhs: &FunctionCall) -> bool {
        match self {
            Expression::Call(call) => &**call == rhs,
            _ => false
        }
    }
}

impl PartialEq<Variable> for Expression {
    fn eq(&self, rhs: &Variable) -> bool {
        match self {
            Expression::Variable(var) => var == rhs,
            _ => false
        }
    }
}

impl PartialEq<Identifier> for Expression {
    fn eq(&self, rhs: &Identifier) -> bool {
        match self {
            Expression::Variable(var) => var == rhs,
            _ => false
        }
    }
}

impl PartialEq<BuiltInIdentifier> for Expression {
    fn eq(&self, rhs: &BuiltInIdentifier) -> bool {
        *self == Identifier::BuiltIn(*rhs)
    }
}

impl PartialEq<Name> for Expression {
    fn eq(&self, rhs: &Name) -> bool {
        match self {
            Expression::Variable(var) => var == rhs,
            _ => false
        }
    }
}

impl PartialEq<Literal> for Expression {
    fn eq(&self, rhs: &Literal) -> bool {
        match self {
            Expression::Literal(lit) => lit == rhs,
            _ => false
        }
    }
}

impl AstNodeFuncs for Expression {
    fn pos(&self) -> &TextPosition {
        match self {
            Expression::Call(call) => call.pos(),
            Expression::Variable(var) => var.pos(),
            Expression::Literal(lit) => lit.pos()
        }
    }
}

impl AstNode for Expression {}

impl Expression {

    #[cfg(test)]
    pub fn call(f: Expression, params: Vec<Expression>) -> Expression {
        Self::from(FunctionCall {
            pos: TextPosition::NONEXISTING,
            function: f,
            parameters: params,
            result_type_cache: RefCell::from(None)
        })
    }

    #[cfg(test)]
    pub fn var(n: &'static str) -> Expression {
        Self::from(Variable {
            pos: TextPosition::NONEXISTING,
            identifier: Identifier::Name(Name::l(n))
        })
    }

    #[cfg(test)]
    pub fn lit(x: i32) -> Expression {
        Self::from(Literal {
            pos: TextPosition::NONEXISTING,
            value: x,
            literal_type: Type::scalar_type(PrimitiveType::Int)
        })
    }
}

#[test]
fn test_expression_preorder_search() {
    let expr = Expression::call(
        Expression::var("foo"),
        vec![Expression::lit(1), Expression::lit(2)]
    );
    let mut found: Vec<&Expression> = Vec::new();
    let mut visitor = |expr| {
        found.push(expr);
        Ok(())
    };
    expr.traverse_preorder(&mut visitor).unwrap();
    assert_eq!(expr, *found[0]);
    assert_eq!(Expression::var("foo"), *found[1]);
    assert_eq!(Expression::lit(1), *found[2]);
    assert_eq!(Expression::lit(2), *found[3]);
}

#[test]
fn test_preorder_search_abort() {
    let expr = Expression::call(
        Expression::var("foo"),
        vec![
            Expression::call(
                Expression::var("bar"), 
                vec![Expression::lit(0)]
            )
        ]
    );
    let mut counter = 0;
    let mut visitor = |expr: &Expression| {
        counter += 1;
        match expr {
            Expression::Call(c) if c.function == Name::l("bar") => DONT_RECURSE,
            _ => Ok(())
        }
    };
    expr.traverse_preorder(&mut visitor).unwrap();
    assert_eq!(3, counter);
}

#[test]
fn test_names_iter() {
    let expr = Expression::call(
        Expression::var("foo"), 
        vec![
            Expression::var("bar"),
            Expression::var("baz")
        ]
    );
    assert_eq!(vec![
        Name::l("baz"),
        Name::l("bar"),
        Name::l("foo")
    ], expr.names().cloned().collect::<Vec<_>>());
}