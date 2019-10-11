use super::ast::*;
use super::super::lexer::error::CompileError;

type VisitorReturnType = Result<(), CompileError>;

pub trait Visitable<T: ?Sized> {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a T) -> VisitorReturnType, T: 'a;
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
        match self.get_kind() {
            StmtKind::Declaration(declaration) => declaration.iterate(f), 
            StmtKind::Assignment(assignment) => assignment.iterate(f), 
            StmtKind::Expr(expr) => expr.iterate(f), 
            StmtKind::If(if_node) => if_node.iterate(f), 
            StmtKind::While(while_node) => while_node.iterate(f), 
            StmtKind::Block(block) => block.iterate(f), 
            StmtKind::Return(return_node) => return_node.iterate(f)
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
