use super::super::lexer::tokens::{Identifier, Literal};
use super::super::lexer::position::TextPosition;

use std::fmt::Debug;
use std::any::Any;

pub type AstVec<T> = Vec<Box<T>>;
type Annotation = TextPosition;

pub trait Node : Debug + Any {
	fn get_annotation(&self) -> &Annotation;
	fn get_annotation_mut(&mut self) -> &mut Annotation;
}

#[derive(Debug)]
pub struct FunctionNode {
	pub annotation: Annotation,
	pub ident: Identifier,
	pub params: AstVec<ParameterNode>,
	pub result: Box<TypeNode>,
	pub stmts: Box<StmtsNode>
}

impl FunctionNode {
	pub fn new(annotation: Annotation, ident: Identifier, params: AstVec<ParameterNode>, result: Box<TypeNode>, stmts: Box<StmtsNode>) -> Self {
		FunctionNode {
			annotation, ident, params, result, stmts
		}
	}
}

#[derive(Debug)]
pub struct ParameterNode {
	pub annotation: Annotation,
	pub ident: Identifier,
	pub param_type: Box<TypeNode>
}

impl ParameterNode {
	pub fn new(annotation: Annotation, ident: Identifier, param_type: Box<TypeNode>) -> Self {
		ParameterNode {
			annotation, ident, param_type
		}
	}
}

#[derive(Debug)]
pub struct StmtsNode {
	pub annotation: Annotation,
	pub stmts: AstVec<StmtNode>
}

impl StmtsNode {
	pub fn new(annotation: Annotation, stmts: AstVec<StmtNode>) -> Self {
		StmtsNode {
			annotation, stmts
		}
	}
}

pub trait StmtNode : Node {
	fn get_type<'a>(&'a self) -> StmtType<'a>;
}

pub enum StmtType<'a> {
	Declaration(&'a DeclarationNode), 
	Assignment(&'a AssignmentNode), 
	Expr(&'a ExprStmtNode), 
	If(&'a IfNode), 
	While(&'a WhileNode), 
	Block(&'a BlockNode), 
	Return(&'a ReturnNode)
}

#[derive(Debug)]
pub struct DeclarationNode {
	annotation: Annotation,
	variable_type: Box<TypeNode>,
	ident: Identifier,
	expr: Option<Box<ExprNode>>
}

impl DeclarationNode {
	pub fn new(annotation: Annotation, variable_type: Box<TypeNode>, ident: Identifier, expr: Option<Box<ExprNode>>) -> Self {
		DeclarationNode {
			annotation, variable_type, ident, expr
		}
	}
}

impl StmtNode for DeclarationNode {
	fn get_type<'a>(&'a self) -> StmtType<'a> {
		StmtType::Declaration(&self)
	}
}

#[derive(Debug)]
pub struct AssignmentNode {
	annotation: Annotation,
	assignee: Box<ExprNode>,
	expr: Box<ExprNode>
}

impl AssignmentNode {
	pub fn new(annotation: Annotation, assignee: Box<ExprNode>, expr: Box<ExprNode>) -> Self {
		AssignmentNode {
			annotation, assignee, expr
		}
	}
}

impl StmtNode for AssignmentNode {
	fn get_type<'a>(&'a self) -> StmtType<'a> {
		StmtType::Assignment(&self)
	}
}

#[derive(Debug)]
pub struct ExprStmtNode {
	annotation: Annotation,
	expr: Box<ExprNode>
}

impl ExprStmtNode {
	pub fn new(annotation: Annotation, expr: Box<ExprNode>) -> Self {
		ExprStmtNode {
			annotation, expr
		}
	}
}

impl StmtNode for ExprStmtNode {
	fn get_type<'a>(&'a self) -> StmtType<'a> {
		StmtType::Expr(&self)
	}
}

#[derive(Debug)]
pub struct IfNode {
	annotation: Annotation,
	condition: Box<ExprNode>,
	block: Box<StmtsNode>
}

impl IfNode {
	pub fn new(annotation: Annotation, condition: Box<ExprNode>, block: Box<StmtsNode>) -> Self {
		IfNode {
			annotation, condition, block
		}
	}
}

impl StmtNode for IfNode {
	fn get_type<'a>(&'a self) -> StmtType<'a> {
		StmtType::If(&self)
	}
}

#[derive(Debug)]
pub struct WhileNode {
	annotation: Annotation,
	condition: Box<ExprNode>,
	block: Box<StmtsNode>
}

impl WhileNode {
	pub fn new(annotation: Annotation, condition: Box<ExprNode>, block: Box<StmtsNode>) -> Self {
		WhileNode {
			annotation, condition, block
		}
	}
}

impl StmtNode for WhileNode {
	fn get_type<'a>(&'a self) -> StmtType<'a> {
		StmtType::While(&self)
	}
}

#[derive(Debug)]
pub struct BlockNode {
	annotation: Annotation,
	block: Box<StmtsNode>
}

impl BlockNode {
	pub fn new(annotation: Annotation, block: Box<StmtsNode>) -> Self {
		BlockNode {
			annotation, block
		}
	}
}

impl StmtNode for BlockNode {
	fn get_type<'a>(&'a self) -> StmtType<'a> {
		StmtType::Block(&self)
	}
}

#[derive(Debug)]
pub struct ReturnNode {
	annotation: Annotation,
	expr: Box<ExprNode>
}

impl ReturnNode {
	pub fn new(annotation: Annotation, expr: Box<ExprNode>) -> Self {
		ReturnNode {
			annotation, expr
		}
	}
}

impl StmtNode for ReturnNode {
	fn get_type<'a>(&'a self) -> StmtType<'a> {
		StmtType::Return(&self)
	}
}

pub trait TypeNode : Node {
	fn get_type<'a>(&'a self) -> TypeType<'a>;
}

pub enum TypeType<'a> {
	Array(&'a ArrTypeNode), Void(&'a VoidTypeNode)
}

#[derive(Debug)]
pub struct ArrTypeNode {
	annotation: Annotation,
	base_type: Box<BaseTypeNode>,
	dims: u8
}

impl ArrTypeNode {
	pub fn new(annotation: Annotation, base_type: Box<BaseTypeNode>, dims: u8) -> Self {
		ArrTypeNode {
			annotation, base_type, dims
		}
	}
}

impl TypeNode for ArrTypeNode {
	fn get_type<'a>(&'a self) -> TypeType<'a> {
		TypeType::Array(&self)
	}
}

#[derive(Debug)]
pub struct VoidTypeNode {
	annotation: Annotation
}

impl VoidTypeNode {
	pub fn new(annotation: Annotation) -> Self {
		VoidTypeNode {
			annotation
		}
	}
}

impl TypeNode for VoidTypeNode {
	fn get_type<'a>(&'a self) -> TypeType<'a> {
		TypeType::Void(&self)
	}
}

pub type ExprNode = ExprNodeLvlOr;

#[derive(Debug)]
pub struct ExprNodeLvlOr {
	annotation: Annotation,
	head: Box<ExprNodeLvlAnd>,
	tail: AstVec<OrPartNode>
}

impl ExprNodeLvlOr {
	pub fn new(annotation: Annotation, head: Box<ExprNodeLvlAnd>, tail: AstVec<OrPartNode>) -> Self {
		ExprNodeLvlOr {
			annotation, head, tail
		}
	}
}

#[derive(Debug)]
pub struct OrPartNode {
	annotation: Annotation,
	expr: Box<ExprNodeLvlAnd>
}

impl OrPartNode {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAnd>) -> Self {
		OrPartNode {
			annotation, expr
		}
	}
}

#[derive(Debug)]
pub struct ExprNodeLvlAnd {
	annotation: Annotation,
	head: Box<ExprNodeLvlCmp>,
	tail: AstVec<AndPartNode>
}

impl ExprNodeLvlAnd {
	pub fn new(annotation: Annotation, head: Box<ExprNodeLvlCmp>, tail: AstVec<AndPartNode>) -> Self {
		ExprNodeLvlAnd {
			annotation, head, tail
		}
	}
}

#[derive(Debug)]
pub struct AndPartNode {
	annotation: Annotation,
	expr: Box<ExprNodeLvlCmp>
}

impl AndPartNode {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlCmp>) -> Self {
		AndPartNode {
			annotation, expr
		}
	}
}

#[derive(Debug)]
pub struct ExprNodeLvlCmp {
	annotation: Annotation, 
	head: Box<ExprNodeLvlAdd>,
	tail: AstVec<CmpPartNode>
}

impl ExprNodeLvlCmp {
	pub fn new(annotation: Annotation, head: Box<ExprNodeLvlAdd>, tail: AstVec<CmpPartNode>) -> Self {
		ExprNodeLvlCmp {
			annotation, head, tail
		}
	}
}

pub trait CmpPartNode : Node {
	fn get_type<'a>(&'a self) -> CmpPartType<'a>;
	fn get_expr(&self) -> &ExprNodeLvlAdd;
}

pub enum CmpPartType<'a> {
	Eq(&'a CmpPartNodeEq), 
	Neq(&'a CmpPartNodeNeq), 
	Leq(&'a CmpPartNodeLeq), 
	Geq(&'a CmpPartNodeGeq), 
	Ls(&'a CmpPartNodeLs), 
	Gt(&'a CmpPartNodeGt)
}

#[derive(Debug)]
pub struct CmpPartNodeEq {
	annotation: Annotation,
	expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeEq {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeEq {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeEq {
	fn get_type<'a>(&'a self) -> CmpPartType<'a> {
		CmpPartType::Eq(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct CmpPartNodeNeq {
	annotation: Annotation,
	expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeNeq {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeNeq {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeNeq {
	fn get_type<'a>(&'a self) -> CmpPartType<'a> {
		CmpPartType::Neq(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct CmpPartNodeLeq {
	annotation: Annotation,
	expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeLeq {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeLeq {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeLeq {
	fn get_type<'a>(&'a self) -> CmpPartType<'a> {
		CmpPartType::Leq(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct CmpPartNodeGeq {
	annotation: Annotation,
	expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeGeq {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeGeq {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeGeq {
	fn get_type<'a>(&'a self) -> CmpPartType<'a> {
		CmpPartType::Geq(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct CmpPartNodeLs {
	annotation: Annotation,
	expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeLs {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeLs {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeLs {
	fn get_type<'a>(&'a self) -> CmpPartType<'a> {
		CmpPartType::Ls(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct CmpPartNodeGt {
	annotation: Annotation,
	expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeGt {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeGt {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeGt {
	fn get_type<'a>(&'a self) -> CmpPartType<'a> {
		CmpPartType::Gt(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct ExprNodeLvlAdd {
	annotation: Annotation,
	head: Box<ExprNodeLvlMult>,
	tail: AstVec<SumPartNode>
}

impl ExprNodeLvlAdd {
	pub fn new(annotation: Annotation, head: Box<ExprNodeLvlMult>, tail: AstVec<SumPartNode>) -> Self {
		ExprNodeLvlAdd {
			annotation, head, tail
		}
	}
}

pub trait SumPartNode : Node {
	fn get_type<'a>(&'a self) -> SumPartType<'a>;
	fn get_expr(&self) -> &ExprNodeLvlMult;
}

pub enum SumPartType<'a> {
	Add(&'a SumPartNodeAdd), Subtract(&'a SumPartNodeSub)
}

#[derive(Debug)]
pub struct SumPartNodeAdd {
	annotation: Annotation,
	expr: Box<ExprNodeLvlMult>
}

impl SumPartNodeAdd {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlMult>) -> Self {
		SumPartNodeAdd {
			annotation, expr
		}
	}
}

impl SumPartNode for SumPartNodeAdd {
	fn get_type<'a>(&'a self) -> SumPartType<'a> {
		SumPartType::Add(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlMult {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct SumPartNodeSub {
	annotation: Annotation,
	expr: Box<ExprNodeLvlMult>
}

impl SumPartNodeSub {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlMult>) -> Self {
		SumPartNodeSub {
			annotation, expr
		}
	}
}

impl SumPartNode for SumPartNodeSub {
	fn get_type<'a>(&'a self) -> SumPartType<'a> {
		SumPartType::Subtract(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlMult {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct ExprNodeLvlMult {
	annotation: Annotation,
	head: Box<ExprNodeLvlIndex>,
	tail: AstVec<ProductPartNode>
}

impl ExprNodeLvlMult {
	pub fn new(annotation: Annotation, head: Box<ExprNodeLvlIndex>, tail: AstVec<ProductPartNode>) -> Self {
		ExprNodeLvlMult {
			annotation, head, tail
		}
	}
}

pub trait ProductPartNode : Node {
	fn get_type<'a>(&'a self) -> ProductPartType<'a>;
	fn get_expr(&self) -> &ExprNodeLvlIndex;
}

pub enum ProductPartType<'a> {
	Mult(&'a ProductPartNodeMult), Divide(&'a ProductPartNodeDivide)
}

#[derive(Debug)]
pub struct ProductPartNodeMult {
	annotation: Annotation,
	expr: Box<ExprNodeLvlIndex>
}

impl ProductPartNodeMult {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlIndex>) -> Self {
		ProductPartNodeMult {
			annotation, expr
		}
	}
}

impl ProductPartNode for ProductPartNodeMult {
	fn get_type<'a>(&'a self) -> ProductPartType<'a> {
		ProductPartType::Mult(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlIndex {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct ProductPartNodeDivide {
	annotation: Annotation,
	expr: Box<ExprNodeLvlIndex>
}

impl ProductPartNodeDivide {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlIndex>) -> Self {
		ProductPartNodeDivide {
			annotation, expr
		}
	}
}

impl ProductPartNode for ProductPartNodeDivide {
	fn get_type<'a>(&'a self) -> ProductPartType<'a> {
		ProductPartType::Divide(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlIndex {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct ExprNodeLvlIndex {
	annotation: Annotation,
	head: Box<UnaryExprNode>,
	tail: AstVec<IndexPartNode>
}

impl ExprNodeLvlIndex {
	pub fn new(annotation: Annotation, head: Box<UnaryExprNode>, tail: AstVec<IndexPartNode>) -> Self {
		ExprNodeLvlIndex {
			annotation, head, tail
		}
	}
}

#[derive(Debug)]
pub struct IndexPartNode {
	annotation: Annotation,
	expr: Box<ExprNode>
}

impl IndexPartNode {
	pub fn new(annotation: Annotation, expr: Box<ExprNode>) -> Self {
		IndexPartNode {
			annotation, expr
		}
	}
}

pub trait UnaryExprNode : Node {
	fn get_type<'a>(&'a self) -> UnaryExprType<'a>;
}

pub enum UnaryExprType<'a> {
	BracketExpr(&'a BracketExprNode), 
	Literal(&'a LiteralNode), 
	Variable(&'a VariableNode), 
	FunctionCall(&'a FunctionCallNode), 
	NewExpr(&'a NewExprNode)
}

#[derive(Debug)]
pub struct BracketExprNode {
	annotation: Annotation,
	expr: Box<ExprNode>
}

impl BracketExprNode {
	pub fn new(annotation: Annotation, expr: Box<ExprNode>) -> Self {
		BracketExprNode {
			annotation, expr
		}
	}
}

impl UnaryExprNode for BracketExprNode {
	fn get_type<'a>(&'a self) -> UnaryExprType<'a> {
		UnaryExprType::BracketExpr(&self)
	}
}

#[derive(Debug)]
pub struct LiteralNode {
	annotation: Annotation,
	literal: Literal
}

impl LiteralNode {
	pub fn new(annotation: Annotation, literal: Literal) -> Self {
		LiteralNode {
			annotation, literal
		}
	}
}

impl UnaryExprNode for LiteralNode {
	fn get_type<'a>(&'a self) -> UnaryExprType<'a> {
		UnaryExprType::Literal(&self)
	}
}

#[derive(Debug)]
pub struct VariableNode {
	annotation: Annotation,
	identifier: Identifier
}

impl VariableNode {
	pub fn new(annotation: Annotation, identifier: Identifier) -> Self {
		VariableNode {
			annotation, identifier
		}
	}
}

impl UnaryExprNode for VariableNode {
	fn get_type<'a>(&'a self) -> UnaryExprType<'a> {
		UnaryExprType::Variable(&self)
	}
}

#[derive(Debug)]
pub struct FunctionCallNode {
	annotation: Annotation,
	function: Identifier,
	params: AstVec<ExprNode>
}

impl FunctionCallNode {
	pub fn new(annotation: Annotation, function: Identifier, params: AstVec<ExprNode>) -> Self {
		FunctionCallNode {
			annotation, function, params
		}
	}
}

impl UnaryExprNode for FunctionCallNode {
	fn get_type<'a>(&'a self) -> UnaryExprType<'a> {
		UnaryExprType::FunctionCall(&self)
	}
}

#[derive(Debug)]
pub struct NewExprNode {
	annotation: Annotation,
	base_type: Box<BaseTypeNode>,
	dimensions: AstVec<IndexPartNode>
}

impl NewExprNode {
	pub fn new(annotation: Annotation, base_type: Box<BaseTypeNode>, dimensions: AstVec<IndexPartNode>) -> Self {
		NewExprNode {
			annotation, base_type, dimensions
		}
	}
}

impl UnaryExprNode for NewExprNode {
	fn get_type<'a>(&'a self) -> UnaryExprType<'a> {
		UnaryExprType::NewExpr(&self)
	}
}

pub trait BaseTypeNode : Node {
	fn get_type(&self) -> BaseTypeType;
}

pub enum BaseTypeType {
	Int
}

#[derive(Debug)]
pub struct IntTypeNode {
	annotation: Annotation
}

impl IntTypeNode {
	pub fn new(annotation: Annotation) -> Self {
		IntTypeNode {
			annotation
		}
	}
}

impl BaseTypeNode for IntTypeNode {
	fn get_type(&self) -> BaseTypeType {
		BaseTypeType::Int
	}
}

macro_rules! impl_node {
	($nodetype:ty) => {
		impl Node for $nodetype {
			fn get_annotation(&self) -> &Annotation {
				&self.annotation
			}

			fn get_annotation_mut(&mut self) -> &mut Annotation {
				&mut self.annotation
			}
		}
	};
}

impl_node!(FunctionNode);
impl_node!(ParameterNode);
impl_node!(StmtsNode);
impl_node!(DeclarationNode);
impl_node!(AssignmentNode);
impl_node!(ExprStmtNode);
impl_node!(IfNode);
impl_node!(WhileNode);
impl_node!(BlockNode);
impl_node!(ReturnNode);
impl_node!(ArrTypeNode);
impl_node!(VoidTypeNode);
impl_node!(ExprNodeLvlOr);
impl_node!(OrPartNode);
impl_node!(ExprNodeLvlAnd);
impl_node!(AndPartNode);
impl_node!(ExprNodeLvlCmp);
impl_node!(CmpPartNodeEq);
impl_node!(CmpPartNodeNeq);
impl_node!(CmpPartNodeLeq);
impl_node!(CmpPartNodeGeq);
impl_node!(CmpPartNodeLs);
impl_node!(CmpPartNodeGt);
impl_node!(ExprNodeLvlAdd);
impl_node!(SumPartNodeAdd);
impl_node!(SumPartNodeSub);
impl_node!(ExprNodeLvlMult);
impl_node!(ProductPartNodeMult);
impl_node!(ProductPartNodeDivide);
impl_node!(ExprNodeLvlIndex);
impl_node!(IndexPartNode);
impl_node!(BracketExprNode);
impl_node!(LiteralNode);
impl_node!(VariableNode);
impl_node!(FunctionCallNode);
impl_node!(NewExprNode);
impl_node!(IntTypeNode);