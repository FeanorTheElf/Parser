use super::ast::*;
use super::super::lexer::error::CompileError;

#[cfg(test)]
use super::super::parser::Parse;
#[cfg(test)]
use super::super::lexer::lexer::lex;
#[cfg(test)]
use super::super::lexer::position::TextPosition;
#[cfg(test)]
use super::super::lexer::tokens::Identifier;

type VisitorReturnType = Result<(), CompileError>;
type VisitorFunctionType<'a, 'b, T> = dyn 'b + FnMut(&'a T) -> VisitorReturnType;

pub trait Visitable<T: ?Sized> {
    fn iterate<'a, 'b>(&'a self, f: &'b mut VisitorFunctionType<'a, 'b, T>) -> VisitorReturnType;
}

impl Visitable<dyn UnaryExprNode> for FunctionNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        match self.implementation.get_concrete() {
            ConcreteFunctionImplementationRef::Implemented(ref implementation) => implementation.iterate(f),
            ConcreteFunctionImplementationRef::Native(ref native) => Ok(())
        }
    }
}

impl Visitable<dyn UnaryExprNode> for ImplementedFunctionNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.body.iterate(f)
    }
}

impl Visitable<dyn UnaryExprNode> for BlockNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        for stmt in self.stmts.iter() {
            stmt.iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for dyn StmtNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        match self.get_concrete() {
            ConcreteStmtRef::Declaration(declaration) => declaration.iterate(f), 
            ConcreteStmtRef::Assignment(assignment) => assignment.iterate(f), 
            ConcreteStmtRef::Expr(expr) => expr.iterate(f), 
            ConcreteStmtRef::If(if_node) => if_node.iterate(f), 
            ConcreteStmtRef::While(while_node) => while_node.iterate(f), 
            ConcreteStmtRef::Block(block) => block.iterate(f), 
            ConcreteStmtRef::Return(return_node) => return_node.iterate(f)
        }
    }
}

impl Visitable<dyn UnaryExprNode> for VariableDeclarationNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.expr.iterate(f)
    }
}

impl Visitable<dyn UnaryExprNode> for AssignmentNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.assignee.iterate(f)?;
        return self.expr.iterate(f);
    }
}

impl Visitable<dyn UnaryExprNode> for ExprStmtNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.expr.iterate(f)
    }
}

impl Visitable<dyn UnaryExprNode> for IfNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.condition.iterate(f)?;
        return self.block.iterate(f);
    }
}

impl Visitable<dyn UnaryExprNode> for WhileNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.condition.iterate(f)?;
        return self.block.iterate(f);
    }
}

impl Visitable<dyn UnaryExprNode> for ReturnNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.expr.iterate(f)
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlOr {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlAnd {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlCmp {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlAdd {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlMult {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlIndex {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        f(&*self.head)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}

type TransformResultType<T> = Result<Box<T>, CompileError>;

pub trait Transformable<T>
    where T: ?Sized
{
    fn transform<F>(&mut self, f: &mut F) -> VisitorReturnType
        where F: FnMut(Box<T>) -> TransformResultType<T>;
}

impl Transformable<dyn StmtNode> for FunctionNode {
    fn transform<F>(&mut self, f: &mut F) -> VisitorReturnType
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        self.implementation.transform(f)?;
        return Ok(());
    }
}

impl Transformable<dyn StmtNode> for dyn FunctionImplementationNode {
    fn transform<F>(&mut self, f: &mut F) -> VisitorReturnType
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        match self.get_mut_concrete() {
            ConcreteFunctionImplementationMut::Implemented(implementation) => implementation.transform(f)?,
            ConcreteFunctionImplementationMut::Native(native) => native.transform(f)?
        };
        return Ok(());
    }
}

impl Transformable<dyn StmtNode> for ImplementedFunctionNode {
    fn transform<F>(&mut self, f: &mut F) -> VisitorReturnType
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        self.body.transform(f)?;
        return Ok(());
    }
}

impl Transformable<dyn StmtNode> for NativeFunctionNode {
    fn transform<F>(&mut self, f: &mut F) -> VisitorReturnType
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        Ok(())
    }
}

impl Transformable<dyn StmtNode> for BlockNode {
    fn transform<F>(&mut self, f: &mut F) -> VisitorReturnType
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        let stmts: Result<AstVec<dyn StmtNode>, CompileError> = self.stmts.drain(..).map(f).collect();
        self.stmts = stmts?;
        return Ok(());
    }
}

#[cfg(test)]
fn rek_transform(mut stmt: Box<dyn StmtNode>) -> Result<Box<dyn StmtNode>, CompileError> {
    match stmt.get_mut_concrete() {
        ConcreteStmtMut::Block(stmts) => stmts.transform(&mut rek_transform)?,
        ConcreteStmtMut::If(stmt) => stmt.block.transform(&mut rek_transform)?,
        ConcreteStmtMut::While(stmt) => stmt.block.transform(&mut rek_transform)?,
        _ => ()
    };

    let global_function_call = ExprNode::parse(&mut lex("global()")).unwrap();
    let global_function_call_stmt = Box::new(ExprStmtNode::new(TextPosition::create(0, 0), global_function_call));

    let stmts = vec![stmt, global_function_call_stmt.clone()];
    let result: Box<dyn StmtNode> = Box::new(BlockNode::new(TextPosition::create(0, 0), stmts));
    return Ok(result);
}

#[test]
fn test_transform_function() {
    let function = FunctionNode::parse(&mut lex("fn min(a: int, b: int,): int {
        if (a < b) {
            return a;
        }
        return b;
    }")).unwrap();
    let expected_function = *FunctionNode::parse(&mut lex("fn min(a: int, b: int,): int {
        {
            if (a < b) {
                {
                    return a;
                    global();
                }
            }
            global();
        }
        {
            return b;
            global();
        }
    }")).unwrap();

    let mut transformed_function = *function.clone();
    transformed_function.transform(&mut rek_transform).unwrap();
    assert_eq!(expected_function, transformed_function);
}