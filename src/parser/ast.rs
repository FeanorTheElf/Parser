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
	annotation: Annotation,
	pub ident: Identifier,
	pub params: AstVec<ParameterNode>,
	pub result: Box<dyn TypeNode>,
	pub body: Box<StmtsNode>
}

impl FunctionNode {
	pub fn new(annotation: Annotation, ident: Identifier, params: AstVec<ParameterNode>, result: Box<dyn TypeNode>, body: Box<StmtsNode>) -> Self {
		FunctionNode {
			annotation, ident, params, result, body
		}
	}
}

#[derive(Debug)]
pub struct ParameterNode {
	annotation: Annotation,
	pub ident: Identifier,
	pub param_type: Box<dyn TypeNode>
}

impl ParameterNode {
	pub fn new(annotation: Annotation, ident: Identifier, param_type: Box<dyn TypeNode>) -> Self {
		ParameterNode {
			annotation, ident, param_type
		}
	}
}

#[derive(Debug)]
pub struct StmtsNode {
	annotation: Annotation,
	pub stmts: AstVec<dyn StmtNode>
}

impl StmtsNode {
	pub fn new(annotation: Annotation, stmts: AstVec<dyn StmtNode>) -> Self {
		StmtsNode {
			annotation, stmts
		}
	}
}

pub trait StmtNode : Node {
	fn get_kind<'a>(&'a self) -> StmtKind<'a>;
}

pub enum StmtKind<'a> {
	Declaration(&'a VariableDeclarationNode), 
	Assignment(&'a AssignmentNode), 
	Expr(&'a ExprStmtNode), 
	If(&'a IfNode), 
	While(&'a WhileNode), 
	Block(&'a BlockNode), 
	Return(&'a ReturnNode)
}

#[derive(Debug)]
pub struct VariableDeclarationNode {
	annotation: Annotation,
	pub variable_type: Box<dyn TypeNode>,
	pub ident: Identifier,
	pub expr: Box<ExprNode>
}

impl VariableDeclarationNode {
	pub fn new(annotation: Annotation, ident: Identifier, variable_type: Box<dyn TypeNode>, expr: Box<ExprNode>) -> Self {
		VariableDeclarationNode {
			annotation, variable_type, ident, expr
		}
	}
}

impl StmtNode for VariableDeclarationNode {
	fn get_kind<'a>(&'a self) -> StmtKind<'a> {
		StmtKind::Declaration(&self)
	}
}

#[derive(Debug)]
pub struct AssignmentNode {
	annotation: Annotation,
	pub assignee: Box<ExprNode>,
	pub expr: Box<ExprNode>
}

impl AssignmentNode {
	pub fn new(annotation: Annotation, assignee: Box<ExprNode>, expr: Box<ExprNode>) -> Self {
		AssignmentNode {
			annotation, assignee, expr
		}
	}
}

impl StmtNode for AssignmentNode {
	fn get_kind<'a>(&'a self) -> StmtKind<'a> {
		StmtKind::Assignment(&self)
	}
}

#[derive(Debug)]
pub struct ExprStmtNode {
	annotation: Annotation,
	pub expr: Box<ExprNode>
}

impl ExprStmtNode {
	pub fn new(annotation: Annotation, expr: Box<ExprNode>) -> Self {
		ExprStmtNode {
			annotation, expr
		}
	}
}

impl StmtNode for ExprStmtNode {
	fn get_kind<'a>(&'a self) -> StmtKind<'a> {
		StmtKind::Expr(&self)
	}
}

#[derive(Debug)]
pub struct IfNode {
	annotation: Annotation,
	pub condition: Box<ExprNode>,
	pub block: Box<StmtsNode>
}

impl IfNode {
	pub fn new(annotation: Annotation, condition: Box<ExprNode>, block: Box<StmtsNode>) -> Self {
		IfNode {
			annotation, condition, block
		}
	}
}

impl StmtNode for IfNode {
	fn get_kind<'a>(&'a self) -> StmtKind<'a> {
		StmtKind::If(&self)
	}
}

#[derive(Debug)]
pub struct WhileNode {
	annotation: Annotation,
	pub condition: Box<ExprNode>,
	pub block: Box<StmtsNode>
}

impl WhileNode {
	pub fn new(annotation: Annotation, condition: Box<ExprNode>, block: Box<StmtsNode>) -> Self {
		WhileNode {
			annotation, condition, block
		}
	}
}

impl StmtNode for WhileNode {
	fn get_kind<'a>(&'a self) -> StmtKind<'a> {
		StmtKind::While(&self)
	}
}

#[derive(Debug)]
pub struct BlockNode {
	annotation: Annotation,
	pub block: Box<StmtsNode>
}

impl BlockNode {
	pub fn new(annotation: Annotation, block: Box<StmtsNode>) -> Self {
		BlockNode {
			annotation, block
		}
	}
}

impl StmtNode for BlockNode {
	fn get_kind<'a>(&'a self) -> StmtKind<'a> {
		StmtKind::Block(&self)
	}
}

#[derive(Debug)]
pub struct ReturnNode {
	annotation: Annotation,
	pub expr: Box<ExprNode>
}

impl ReturnNode {
	pub fn new(annotation: Annotation, expr: Box<ExprNode>) -> Self {
		ReturnNode {
			annotation, expr
		}
	}
}

impl StmtNode for ReturnNode {
	fn get_kind<'a>(&'a self) -> StmtKind<'a> {
		StmtKind::Return(&self)
	}
}

pub trait TypeNode : Node {
	fn get_kind<'a>(&'a self) -> TypeKind<'a>;
}

pub enum TypeKind<'a> {
	Array(&'a ArrTypeNode), Void(&'a VoidTypeNode)
}

#[derive(Debug)]
pub struct ArrTypeNode {
	annotation: Annotation,
	pub base_type: Box<dyn BaseTypeNode>,
	pub dims: u8
}

impl ArrTypeNode {
	pub fn new(annotation: Annotation, base_type: Box<dyn BaseTypeNode>, dims: u8) -> Self {
		ArrTypeNode {
			annotation, base_type, dims
		}
	}

	pub fn get_base_type(&self) -> &dyn BaseTypeNode {
		&*self.base_type
	}

	pub fn get_dims(&self) -> u32 {
		self.dims as u32
	}
}

impl TypeNode for ArrTypeNode {
	fn get_kind<'a>(&'a self) -> TypeKind<'a> {
		TypeKind::Array(&self)
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
	fn get_kind<'a>(&'a self) -> TypeKind<'a> {
		TypeKind::Void(&self)
	}
}

pub type ExprNode = ExprNodeLvlOr;

#[derive(Debug)]
pub struct ExprNodeLvlOr {
	annotation: Annotation,
	pub head: Box<ExprNodeLvlAnd>,
	pub tail: AstVec<OrPartNode>
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
	pub expr: Box<ExprNodeLvlAnd>
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
	pub head: Box<ExprNodeLvlCmp>,
	pub tail: AstVec<AndPartNode>
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
	pub expr: Box<ExprNodeLvlCmp>
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
	pub head: Box<ExprNodeLvlAdd>,
	pub tail: AstVec<dyn CmpPartNode>
}

impl ExprNodeLvlCmp {
	pub fn new(annotation: Annotation, head: Box<ExprNodeLvlAdd>, tail: AstVec<dyn CmpPartNode>) -> Self {
		ExprNodeLvlCmp {
			annotation, head, tail
		}
	}
}

pub trait CmpPartNode : Node {
	fn get_kind<'a>(&'a self) -> CmpPartKind<'a>;
	fn get_expr(&self) -> &ExprNodeLvlAdd;
}

pub enum CmpPartKind<'a> {
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
	pub expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeEq {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeEq {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeEq {
	fn get_kind<'a>(&'a self) -> CmpPartKind<'a> {
		CmpPartKind::Eq(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct CmpPartNodeNeq {
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeNeq {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeNeq {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeNeq {
	fn get_kind<'a>(&'a self) -> CmpPartKind<'a> {
		CmpPartKind::Neq(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct CmpPartNodeLeq {
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeLeq {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeLeq {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeLeq {
	fn get_kind<'a>(&'a self) -> CmpPartKind<'a> {
		CmpPartKind::Leq(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct CmpPartNodeGeq {
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeGeq {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeGeq {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeGeq {
	fn get_kind<'a>(&'a self) -> CmpPartKind<'a> {
		CmpPartKind::Geq(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct CmpPartNodeLs {
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeLs {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeLs {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeLs {
	fn get_kind<'a>(&'a self) -> CmpPartKind<'a> {
		CmpPartKind::Ls(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct CmpPartNodeGt {
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeGt {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self {
		CmpPartNodeGt {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeGt {
	fn get_kind<'a>(&'a self) -> CmpPartKind<'a> {
		CmpPartKind::Gt(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlAdd {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct ExprNodeLvlAdd {
	annotation: Annotation,
	pub head: Box<ExprNodeLvlMult>,
	pub tail: AstVec<dyn SumPartNode>
}

impl ExprNodeLvlAdd {
	pub fn new(annotation: Annotation, head: Box<ExprNodeLvlMult>, tail: AstVec<dyn SumPartNode>) -> Self {
		ExprNodeLvlAdd {
			annotation, head, tail
		}
	}
}

pub trait SumPartNode : Node {
	fn get_kind<'a>(&'a self) -> SumPartKind<'a>;
	fn get_expr(&self) -> &ExprNodeLvlMult;
}

pub enum SumPartKind<'a> {
	Add(&'a SumPartNodeAdd), Subtract(&'a SumPartNodeSub)
}

#[derive(Debug)]
pub struct SumPartNodeAdd {
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlMult>
}

impl SumPartNodeAdd {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlMult>) -> Self {
		SumPartNodeAdd {
			annotation, expr
		}
	}
}

impl SumPartNode for SumPartNodeAdd {
	fn get_kind<'a>(&'a self) -> SumPartKind<'a> {
		SumPartKind::Add(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlMult {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct SumPartNodeSub {
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlMult>
}

impl SumPartNodeSub {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlMult>) -> Self {
		SumPartNodeSub {
			annotation, expr
		}
	}
}

impl SumPartNode for SumPartNodeSub {
	fn get_kind<'a>(&'a self) -> SumPartKind<'a> {
		SumPartKind::Subtract(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlMult {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct ExprNodeLvlMult {
	annotation: Annotation,
	pub head: Box<ExprNodeLvlIndex>,
	pub tail: AstVec<dyn ProductPartNode>
}

impl ExprNodeLvlMult {
	pub fn new(annotation: Annotation, head: Box<ExprNodeLvlIndex>, tail: AstVec<dyn ProductPartNode>) -> Self {
		ExprNodeLvlMult {
			annotation, head, tail
		}
	}
}

pub trait ProductPartNode : Node {
	fn get_kind<'a>(&'a self) -> ProductPartKind<'a>;
	fn get_expr(&self) -> &ExprNodeLvlIndex;
}

pub enum ProductPartKind<'a> {
	Mult(&'a ProductPartNodeMult), Divide(&'a ProductPartNodeDivide)
}

#[derive(Debug)]
pub struct ProductPartNodeMult {
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlIndex>
}

impl ProductPartNodeMult {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlIndex>) -> Self {
		ProductPartNodeMult {
			annotation, expr
		}
	}
}

impl ProductPartNode for ProductPartNodeMult {
	fn get_kind<'a>(&'a self) -> ProductPartKind<'a> {
		ProductPartKind::Mult(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlIndex {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct ProductPartNodeDivide {
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlIndex>
}

impl ProductPartNodeDivide {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlIndex>) -> Self {
		ProductPartNodeDivide {
			annotation, expr
		}
	}
}

impl ProductPartNode for ProductPartNodeDivide {
	fn get_kind<'a>(&'a self) -> ProductPartKind<'a> {
		ProductPartKind::Divide(&self)
	}

	fn get_expr(&self) -> &ExprNodeLvlIndex {
		&*self.expr
	}
}

#[derive(Debug)]
pub struct ExprNodeLvlIndex {
	annotation: Annotation,
	pub head: Box<dyn UnaryExprNode>,
	pub tail: AstVec<IndexPartNode>
}

impl ExprNodeLvlIndex {
	pub fn new(annotation: Annotation, head: Box<dyn UnaryExprNode>, tail: AstVec<IndexPartNode>) -> Self {
		ExprNodeLvlIndex {
			annotation, head, tail
		}
	}
}

#[derive(Debug)]
pub struct IndexPartNode {
	annotation: Annotation,
	pub expr: Box<ExprNode>
}

impl IndexPartNode {
	pub fn new(annotation: Annotation, expr: Box<ExprNode>) -> Self {
		IndexPartNode {
			annotation, expr
		}
	}
}

pub trait UnaryExprNode : Node {
	fn get_kind<'a>(&'a self) -> UnaryExprKind<'a>;
}

pub enum UnaryExprKind<'a> {
	BracketExpr(&'a BracketExprNode), 
	Literal(&'a LiteralNode), 
	Variable(&'a VariableNode), 
	FunctionCall(&'a FunctionCallNode), 
	NewExpr(&'a NewExprNode)
}

#[derive(Debug)]
pub struct BracketExprNode {
	annotation: Annotation,
	pub expr: Box<ExprNode>
}

impl BracketExprNode {
	pub fn new(annotation: Annotation, expr: Box<ExprNode>) -> Self {
		BracketExprNode {
			annotation, expr
		}
	}
}

impl UnaryExprNode for BracketExprNode {
	fn get_kind<'a>(&'a self) -> UnaryExprKind<'a> {
		UnaryExprKind::BracketExpr(&self)
	}
}

#[derive(Debug)]
pub struct LiteralNode {
	annotation: Annotation,
	pub literal: Literal
}

impl LiteralNode {
	pub fn new(annotation: Annotation, literal: Literal) -> Self {
		LiteralNode {
			annotation, literal
		}
	}
}

impl UnaryExprNode for LiteralNode {
	fn get_kind<'a>(&'a self) -> UnaryExprKind<'a> {
		UnaryExprKind::Literal(&self)
	}
}

#[derive(Debug)]
pub struct VariableNode {
	annotation: Annotation,
	pub identifier: Identifier
}

impl VariableNode {
	pub fn new(annotation: Annotation, identifier: Identifier) -> Self {
		VariableNode {
			annotation, identifier
		}
	}
}

impl UnaryExprNode for VariableNode {
	fn get_kind<'a>(&'a self) -> UnaryExprKind<'a> {
		UnaryExprKind::Variable(&self)
	}
}

#[derive(Debug)]
pub struct FunctionCallNode {
	annotation: Annotation,
	pub function: Identifier,
	pub params: AstVec<ExprNode>
}

impl FunctionCallNode {
	pub fn new(annotation: Annotation, function: Identifier, params: AstVec<ExprNode>) -> Self {
		FunctionCallNode {
			annotation, function, params
		}
	}
}

impl UnaryExprNode for FunctionCallNode {
	fn get_kind<'a>(&'a self) -> UnaryExprKind<'a> {
		UnaryExprKind::FunctionCall(&self)
	}
}

#[derive(Debug)]
pub struct NewExprNode {
	annotation: Annotation,
	pub base_type: Box<dyn BaseTypeNode>,
	pub dimensions: AstVec<IndexPartNode>
}

impl NewExprNode {
	pub fn new(annotation: Annotation, base_type: Box<dyn BaseTypeNode>, dimensions: AstVec<IndexPartNode>) -> Self {
		NewExprNode {
			annotation, base_type, dimensions
		}
	}
}

impl UnaryExprNode for NewExprNode {
	fn get_kind<'a>(&'a self) -> UnaryExprKind<'a> {
		UnaryExprKind::NewExpr(&self)
	}
}

pub trait BaseTypeNode : Node {
	fn get_kind(&self) -> BaseTypeKind;
}

pub enum BaseTypeKind {
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
	fn get_kind(&self) -> BaseTypeKind {
		BaseTypeKind::Int
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
impl_node!(VariableDeclarationNode);
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