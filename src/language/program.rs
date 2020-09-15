use super::error::{CompileError, ErrorType};
use super::identifier::{BuiltInIdentifier, Identifier, Name};
use super::position::{TextPosition, BEGIN};
use super::types::{Type, FunctionType, TypePtr, TypeVec};
use super::AstNode;

use super::super::util::cmp::Comparing;
use super::super::util::dyn_lifetime::*;

use std::cell::Ref;

#[derive(Debug)]
pub struct Program {
    pub items: Vec<Box<Function>>,
    pub types: TypeVec
}

impl Program {
    pub fn lifetime<'a>(&'a self) -> Lifetime<'a> {
        self.types.get_lifetime()
    }

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

pub trait Statement: AstNode {
    fn dyn_clone(&self) -> Box<dyn Statement>;

    ///
    /// Iterates over all blocks contained directly in this statement.
    /// 
    /// # Details
    /// This will not yield nested blocks. If this statement is a block itself, it will only return itself.
    /// The order in which the blocks are returned is unspecified.
    /// 
    /// # Example
    /// ```
    /// let a = Block::parse(&mut fragment_lex("{{{}}{}}"), &mut TypeVec::new()).unwrap();
    /// assert_eq!(a.subblocks().count(), 1); // returns only the top level block
    /// assert_eq!(a.subblocks().flat_map(|b| b.statements.iter()).flat_map(|s| s.subblocks()).count(), 2); // returns the proper subblocks, also non-nested
    /// ```
    /// 
    fn subblocks<'a>(&'a self)-> Box<(dyn Iterator<Item = &'a Block> + 'a)>;
    ///
    /// See subblocks()
    /// 
    fn subblocks_mut<'a>(&'a mut self)-> Box<(dyn Iterator<Item = &'a mut Block> + 'a)>;
    ///
    /// Iterates over all expressions contained directly in this statement.
    /// 
    /// # Details
    /// This will not yield nested expressions. If this statement is only one expression, only the statement itself will be returned.
    /// The expressions will be returned in the order of execution.
    /// 
    /// # Example
    /// ```
    /// let a = Statement::parse(&mut fragment_lex("a = b(c, 0,);"), &mut TypeVec::new()).unwrap();
    /// assert_eq!(a.expressions().count(), 2); // returns only the top level expressions 'a' and 'b(c, 0,)', in this order
    /// ```
    /// 
    fn expressions<'a>(&'a self)-> Box<(dyn Iterator<Item = &'a Expression> + 'a)>;
    ///
    /// See expressions()
    /// 
    fn expressions_mut<'a>(&'a mut self)-> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)>;

    ///
    /// Iterates over all names contained in this statement.
    /// 
    /// # Details
    /// This will return all names (i.e. user-defined identifiers) in this statement, which includes uses of variables/functions/... and
    /// also declarations. Since names cannot be nested, all names are returned (as opposed to subblocks() and expressions()). The order
    /// in which the names are returned is unspecified.
    ///
    /// # Example
    /// ```
    /// let a = Statement::parse(&mut fragment_lex("let a: int = b(c, 0,);"), &mut TypeVec::new()).unwrap();
    /// assert_eq!(a.names().map(|n| n.name.as_str()).collect::<Vec<_>>(), vec!["a", "c", "b"]);
    /// ```
    /// 
    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(self.expressions().flat_map(|e| e.names()))
    }
    ///
    /// See names()
    /// 
    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(self.expressions_mut().flat_map(|e| e.names_mut()))
    }
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
            for expr in statement.expressions() {
                f(expr);
            }
            for subblock in statement.subblocks() {
                subblock.scan_top_level_expressions(f);
            }
        }
    }
}

pub struct ExpressionVarIter<'a> {
    subtrees: Vec<&'a Expression>
}

impl<'a> Iterator for ExpressionVarIter<'a> {
    type Item = &'a Variable;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.subtrees.pop() {
                None => return None,
                Some(Expression::Call(call)) => {
                    self.subtrees.extend(std::iter::once(&call.function).chain(call.parameters.iter()));
                },
                Some(Expression::Variable(var)) => {
                    return Some(var);
                },
                Some(Expression::Literal(_)) => {}
            }
        }
    }
}

pub struct ExpressionVarIterMut<'a> {
    subtrees: Vec<&'a mut Expression>
}

impl<'a> Iterator for ExpressionVarIterMut<'a> {
    type Item = &'a mut Variable;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.subtrees.pop() {
                None => return None,
                Some(Expression::Call(call)) => {
                    self.subtrees.extend(std::iter::once(&mut call.function).chain(call.parameters.iter_mut()));
                },
                Some(Expression::Variable(var)) => {
                    return Some(var);
                },
                Some(Expression::Literal(_)) => {}
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
                Expression::Literal(_) => unimplemented!(),
                Expression::Variable(var) => {
                    var.identifier == Identifier::BuiltIn(BuiltInIdentifier::FunctionIndex)
                        && call.parameters[0].is_lvalue()
                }
            },
            Expression::Variable(_) => true,
            Expression::Literal(_) => false,
        }
    }

    pub fn variables<'a>(&'a self) -> ExpressionVarIter<'a> {
        ExpressionVarIter {
            subtrees: vec![self]
        }
    }

    pub fn variables_mut<'a>(&'a mut self) -> ExpressionVarIterMut<'a> {
        ExpressionVarIterMut {
            subtrees: vec![self]
        }
    }

    pub fn names<'a>(&'a self) -> impl Iterator<Item = &'a Name> {
        self.variables().filter_map(|v| match &v.identifier {
            Identifier::Name(name) => Some(name),
            Identifier::BuiltIn(_) => None
        })
    }

    pub fn names_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Name> {
        self.variables_mut().filter_map(|v| match &mut v.identifier {
            Identifier::Name(name) => Some(name),
            Identifier::BuiltIn(_) => None
        })
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
    
    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(self))
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(self))
    }

    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl Statement for If {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }
    
    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(&self.condition))
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.condition))
    }

    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(&self.body))
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(&mut self.body))
    }
}

impl Statement for While {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(&self.condition))
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.condition))
    }

    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(&self.body))
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(&mut self.body))
    }
}

impl Statement for Return {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        if let Some(ref val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        if let Some(ref mut val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }
    
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl Statement for Block {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(self))
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(self))
    }
}

impl Statement for LocalVariableDeclaration {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        if let Some(ref val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        if let Some(ref mut val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(std::iter::once(&self.declaration.variable).chain((&self.value).into_iter().flat_map(|e| e.names())))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(std::iter::once(&mut self.declaration.variable).chain((&mut self.value).into_iter().flat_map(|e| e.names_mut())))
    }
}

impl Statement for Assignment {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(&self.assignee).chain(std::iter::once(&self.value)))
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.assignee).chain(std::iter::once(&mut self.value)))
    }

    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl Statement for Label {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(std::iter::once(&self.label))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(std::iter::once(&mut self.label))
    }
}

impl Statement for Goto {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(std::iter::once(&self.target))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(std::iter::once(&mut self.target))
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
