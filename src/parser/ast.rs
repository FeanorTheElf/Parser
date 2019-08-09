use super::super::lexer::tokens::{Identifier, Literal};
use super::super::lexer::position::TextPosition;

type Annotation = TextPosition;

#[derive(Debug)]
pub enum Function {
	Function(Annotation, Box<Identifier>, Vec<ParameterDeclaration>, Box<Type>, Box<Stmts>)
}

#[derive(Debug)]
pub enum ParameterDeclaration {
	ParameterDeclaration(Annotation, Box<Identifier>, Box<Type>)
}

#[derive(Debug)]
pub enum Stmts {
	Stmts(Annotation, Vec<Stmt>)
}

#[derive(Debug)]
pub enum Stmt {
	Declaration(Annotation, Box<Type>, Box<Identifier>, Option<Box<Expr>>),
	Assignment(Annotation, Box<Expr>, Box<Expr>),
	Expr(Annotation, Box<Expr>),
	If(Annotation, Box<Expr>, Box<Stmts>),
	While(Annotation, Box<Expr>, Box<Stmts>),
	Block(Annotation, Box<Stmts>),
	Return(Annotation, Box<Expr>)
}

#[derive(Debug)]
pub enum Type {
	Arr(Annotation, Box<BaseType>, u8),
	Void(Annotation)
}

pub type Expr = ExprLvlOr;

#[derive(Debug)]
pub enum ExprLvlOr {
	Or(Annotation, Box<ExprLvlAnd>, Vec<OrPart>)
}

#[derive(Debug)]
pub enum OrPart {
	Expr(Annotation, Box<ExprLvlAnd>)
}

#[derive(Debug)]
pub enum ExprLvlAnd {
	And(Annotation, Box<ExprLvlCmp>, Vec<AndPart>)
}

#[derive(Debug)]
pub enum AndPart {
	Expr(Annotation, Box<ExprLvlCmp>)
}

#[derive(Debug)]
pub enum ExprLvlCmp {
	Cmp(Annotation, Box<ExprLvlAdd>, Vec<CmpPart>)
}

#[derive(Debug)]
pub enum CmpPart {
	Eq(Annotation, Box<ExprLvlAdd>),
	Neq(Annotation, Box<ExprLvlAdd>),
	Leq(Annotation, Box<ExprLvlAdd>),
	Geq(Annotation, Box<ExprLvlAdd>),
	Ls(Annotation, Box<ExprLvlAdd>),
	Gt(Annotation, Box<ExprLvlAdd>)
}

#[derive(Debug)]
pub enum ExprLvlAdd {
	Add(Annotation, Box<ExprLvlMult>, Vec<AddPart>)
}

#[derive(Debug)]
pub enum AddPart {
	Add(Annotation, Box<ExprLvlMult>),
	Subtract(Annotation, Box<ExprLvlMult>)
}

#[derive(Debug)]
pub enum ExprLvlMult {
	Mult(Annotation, Box<ExprLvlIndex>, Vec<MultPart>)
}

#[derive(Debug)]
pub enum MultPart {
	Mult(Annotation, Box<ExprLvlIndex>),
	Divide(Annotation, Box<ExprLvlIndex>)
}

#[derive(Debug)]
pub enum ExprLvlIndex {
	Index(Annotation, Box<UnaryExpr>, Vec<IndexPart>)
}

#[derive(Debug)]
pub enum IndexPart {
	Expr(Annotation, Box<Expr>)
}

#[derive(Debug)]
pub enum UnaryExpr {
	Brackets(Annotation, Box<Expr>),
	Literal(Annotation, Box<Literal>),
	Variable(Annotation, Box<Identifier>),
	Call(Annotation, Box<Identifier>, Vec<Expr>),
	New(Annotation, Box<BaseType>, Vec<IndexPart>)
}

#[derive(Debug)]
pub enum BaseType {
	Int(Annotation)
}
