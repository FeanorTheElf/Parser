use super::tokens::{Identifier, Literal};

#[derive(Debug)]
pub enum Function {
	Function(Box<Identifier>, Vec<ParameterDeclaration>, Box<Type>, Box<Stmts>)
}

#[derive(Debug)]
pub enum ParameterDeclaration {
	ParameterDeclaration(Box<Identifier>, Box<Type>)
}

#[derive(Debug)]
pub enum Stmts {
	Stmts(Vec<Stmt>)
}

#[derive(Debug)]
pub enum Stmt {
	Declaration(Box<Type>, Box<Identifier>, Option<Box<Expr>>),
	Assignment(Box<Expr>, Box<Expr>),
	Expr(Box<Expr>),
	If(Box<Expr>, Box<Stmts>),
	While(Box<Expr>, Box<Stmts>),
	Block(Box<Stmts>),
	Return(Box<Expr>)
}

#[derive(Debug)]
pub enum Type {
	Arr(Box<BaseType>, u8),
	Void()
}

pub type Expr = ExprLvlOr;

#[derive(Debug)]
pub enum ExprLvlOr {
	Or(Box<ExprLvlAnd>, Vec<OrPart>)
}

#[derive(Debug)]
pub enum OrPart {
	Expr(Box<ExprLvlAnd>)
}

#[derive(Debug)]
pub enum ExprLvlAnd {
	And(Box<ExprLvlCmp>, Vec<AndPart>)
}

#[derive(Debug)]
pub enum AndPart {
	Expr(Box<ExprLvlCmp>)
}

#[derive(Debug)]
pub enum ExprLvlCmp {
	Cmp(Box<ExprLvlAdd>, Vec<CmpPart>)
}

#[derive(Debug)]
pub enum CmpPart {
	Eq(Box<ExprLvlAdd>),
	Neq(Box<ExprLvlAdd>),
	Leq(Box<ExprLvlAdd>),
	Geq(Box<ExprLvlAdd>),
	Ls(Box<ExprLvlAdd>),
	Gt(Box<ExprLvlAdd>)
}

#[derive(Debug)]
pub enum ExprLvlAdd {
	Add(Box<ExprLvlMult>, Vec<AddPart>)
}

#[derive(Debug)]
pub enum AddPart {
	Add(Box<ExprLvlMult>),
	Subtract(Box<ExprLvlMult>)
}

#[derive(Debug)]
pub enum ExprLvlMult {
	Mult(Box<ExprLvlIndex>, Vec<MultPart>)
}

#[derive(Debug)]
pub enum MultPart {
	Mult(Box<ExprLvlIndex>),
	Divide(Box<ExprLvlIndex>)
}

#[derive(Debug)]
pub enum ExprLvlIndex {
	Index(Box<UnaryExpr>, Vec<IndexPart>)
}

#[derive(Debug)]
pub enum IndexPart {
	Expr(Box<Expr>)
}

#[derive(Debug)]
pub enum UnaryExpr {
	Brackets(Box<Expr>),
	Literal(Box<Literal>),
	Variable(Box<Identifier>),
	Call(Box<Identifier>, Vec<Expr>),
	New(Box<BaseType>, Vec<IndexPart>)
}

#[derive(Debug)]
pub enum BaseType {
	Int()
}
