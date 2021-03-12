use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::{Identifier, Name};
use super::ast::*;

use std::cell::Cell;

type Type = ();

#[derive(Debug, PartialEq, Eq)]
pub enum Expression {
    Call(Box<FunctionCall>),
    Variable(Variable),
    Literal(Literal),
}

#[derive(Debug, Eq)]
pub struct FunctionCall {
    pub pos: TextPosition,
    pub function: Expression,
    pub parameters: Vec<Expression>,
    pub result_type_cache: Cell<Option<Type>>
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

#[derive(Debug, Eq)]
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

#[cfg(test)]
impl Expression {

    fn call(f: Expression, params: Vec<Expression>) -> Expression {
        Self::from(FunctionCall {
            pos: TextPosition::NONEXISTING,
            function: f,
            parameters: params,
            result_type_cache: Cell::from(None)
        })
    }

    fn var(n: Name) -> Expression {
        Self::from(Variable {
            pos: TextPosition::NONEXISTING,
            identifier: Identifier::Name(n)
        })
    }

    fn lit(x: i32) -> Expression {
        Self::from(Literal {
            pos: TextPosition::NONEXISTING,
            value: x,
            literal_type: ()
        })
    }
}

#[test]
fn test_expression_preorder_search() {
    let expr = Expression::call(
        Expression::var(Name::l("foo")),
        vec![Expression::lit(1), Expression::lit(2)]
    );
    let mut found: Vec<&Expression> = Vec::new();
    let mut visitor = |expr| {
        found.push(expr);
        Ok(())
    };
    expr.traverse_preorder(&mut visitor).unwrap();
    assert_eq!(expr, *found[0]);
    assert_eq!(Expression::var(Name::l("foo")), *found[1]);
    assert_eq!(Expression::lit(1), *found[2]);
    assert_eq!(Expression::lit(2), *found[3]);
}

#[test]
fn test_preorder_search_abort() {
    let expr = Expression::call(
        Expression::var(Name::l("foo")),
        vec![
            Expression::call(
                Expression::var(Name::l("bar")), 
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
    expr.traverse_preorder(&mut visitor);
    assert_eq!(3, counter);
}