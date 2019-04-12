use super::tokens::{Identifier, Literal};

#[derive(Debug)]
pub enum Func {
	Func(Identifier, Vec<ParamDeclaration>, Type, Stmts)
}

#[derive(Debug)]
pub enum ParamDeclaration {
	ParamDeclaration(Type, Identifier)
}

#[derive(Debug)]
pub enum Stmts {
	Stmts(Vec<Stmt>)
}

#[derive(Debug)]
pub enum Stmt {
	Declaration(Type, Identifier, Option<Expr>),
	Assignment(Expr, Expr),
	Expr(Expr),
	If(Expr, Box<Stmts>),
	While(Expr, Box<Stmts>),
	Block(Box<Stmts>),
	Return(Expr)
}

#[derive(Debug)]
pub enum Type {
	Arr(BaseType, u8),
	Void
}

#[derive(Debug)]
pub enum Expr {
	Mult(ExprMult),
	Sum(Vec<Summand>)
}

#[derive(Debug)]
pub enum Summand {
	Pos(ExprMult),
	Neg(ExprMult)
}

#[derive(Debug)]
pub enum ExprMult {
	Unary(ExprUn),
	Product(Vec<ProductPart>)
}

#[derive(Debug)]
pub enum ProductPart {
	Factor(ExprUn),
	Inverse(ExprUn)
}

#[derive(Debug)]
pub enum ExprUn {
	Basic(BasicExprUn),
	Indexed(Box<ExprUn>, Box<Expr>)
}

#[derive(Debug)]
pub enum BasicExprUn {
	Brackets(Box<Expr>),
	Literal(Literal),
	Variable(Identifier),
	Call(Identifier, Vec<Expr>),
	New(BaseType, Vec<Expr>)
}

#[derive(Debug)]
pub enum BaseType {
	Int
}
