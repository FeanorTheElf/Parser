use super::ast::*;
use super::super::lexer::error::CompileError;

type VisitorReturnType = Result<(), CompileError>;

pub trait Visitable<T: ?Sized> {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a T) -> VisitorReturnType, T: 'a;
}

impl Visitable<dyn UnaryExprNode> for FunctionNode {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        match self.implementation.get_concrete() {
            ConcreteFunctionImplementationRef::Implemented(ref implementation) => implementation.iterate(f),
            ConcreteFunctionImplementationRef::Native(ref native) => Ok(())
        }
    }
}

impl Visitable<dyn UnaryExprNode> for ImplementedFunctionNode {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.body.iterate(f)
    }
}

impl Visitable<dyn UnaryExprNode> for StmtsNode {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        for stmt in self.stmts.iter() {
            stmt.iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for dyn StmtNode {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
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
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.expr.iterate(f)
    }
}

impl Visitable<dyn UnaryExprNode> for AssignmentNode {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.assignee.iterate(f)?;
        return self.expr.iterate(f);
    }
}

impl Visitable<dyn UnaryExprNode> for ExprStmtNode {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.expr.iterate(f)
    }
}

impl Visitable<dyn UnaryExprNode> for IfNode {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.condition.iterate(f)?;
        return self.block.iterate(f);
    }
}

impl Visitable<dyn UnaryExprNode> for WhileNode {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.condition.iterate(f)?;
        return self.block.iterate(f);
    }
}

impl Visitable<dyn UnaryExprNode> for BlockNode {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.block.iterate(f)
    }
}

impl Visitable<dyn UnaryExprNode> for ReturnNode {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.expr.iterate(f)
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlOr {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlAnd {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlCmp {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlAdd {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlMult {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlIndex {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
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
    where T: ?Sized, Self: Sized
{
    fn transform<F>(self, f: &mut F) -> Result<Self, CompileError>
        where F: FnMut(Box<T>) -> TransformResultType<T>;
}

impl Transformable<dyn StmtNode> for Box<FunctionNode> {
    fn transform<F>(mut self, f: &mut F) -> Result<Self, CompileError>
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        self.implementation = self.implementation.transform(f)?;
        return Ok(self);
    }
}

impl Transformable<dyn StmtNode> for Box<dyn FunctionImplementationNode> {
    fn transform<F>(self, f: &mut F) -> Result<Self, CompileError>
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        match self.into_concrete() {
            ConcreteFunctionImplementation::Implemented(implementation) => Ok(implementation.transform(f)?),
            ConcreteFunctionImplementation::Native(native) => Ok(native.transform(f)?)
        }
    }
}

impl Transformable<dyn StmtNode> for Box<ImplementedFunctionNode> {
    fn transform<F>(mut self, f: &mut F) -> Result<Self, CompileError>
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        self.body = self.body.transform(f)?;
        return Ok(self);
    }
}

impl Transformable<dyn StmtNode> for Box<NativeFunctionNode> {
    fn transform<F>(self, f: &mut F) -> Result<Self, CompileError>
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        Ok(self)
    }
}

impl Transformable<dyn StmtNode> for Box<StmtsNode> {
    fn transform<F>(mut self, f: &mut F) -> Result<Self, CompileError>
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        let stmts: Result<AstVec<dyn StmtNode>, CompileError> = self.stmts.drain(..).map(|stmt|stmt.transform(f)).collect();
        self.stmts = stmts?;
        return Ok(self);
    }
}

impl Transformable<dyn StmtNode> for Box<dyn StmtNode> {
    fn transform<F>(self, f: &mut F) -> Result<Self, CompileError>
        where F: FnMut(Box<dyn StmtNode>) -> TransformResultType<dyn StmtNode> 
    {
        f(self)
    }
}

