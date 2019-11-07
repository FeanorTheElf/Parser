use super::super::lexer::tokens::{Identifier, Literal};
use super::super::lexer::position::TextPosition;
use super::super::lexer::error::CompileError;

use std::fmt::Debug;
use std::any::Any;

pub type AstVec<T> = Vec<Box<T>>;
type Annotation = TextPosition;

pub trait Node : Debug + Any {
	fn get_annotation(&self) -> &Annotation;
	fn get_annotation_mut(&mut self) -> &mut Annotation;
	fn dyn_clone_node(&self) -> Box<dyn Node>;
	fn dynamic(&self) -> &dyn Any;
}

#[derive(Debug)]
pub struct FunctionNode {
	annotation: Annotation,
	pub ident: Identifier,
	pub params: AstVec<ParameterNode>,
	pub result: Box<dyn TypeNode>,
	pub implementation: Box<dyn FunctionImplementationNode>
}

impl FunctionNode {
	pub fn new(annotation: Annotation, ident: Identifier, params: AstVec<ParameterNode>, result: Box<dyn TypeNode>, implementation: Box<dyn FunctionImplementationNode>) -> Self {
		FunctionNode {
			annotation, ident, params, result, implementation
		}
	}
}

impl Clone for FunctionNode {
	fn clone(&self) -> Self {
		FunctionNode {
			annotation: self.annotation.clone(),
			ident: self.ident.clone(),
			params: self.params.clone(),
			result: self.result.dyn_clone(),
			implementation: self.implementation.dyn_clone()
		}
	}
}

pub trait FunctionImplementationNode : Node {
	fn get_concrete<'a>(&'a self) -> ConcreteFunctionImplementationRef<'a>;
	fn into_concrete(self: Box<Self>) -> ConcreteFunctionImplementation;
	fn dyn_clone(&self) -> Box<dyn FunctionImplementationNode>;
}

pub enum ConcreteFunctionImplementationRef<'a> {
	Implemented(&'a ImplementedFunctionNode), Native(&'a NativeFunctionNode)
}

pub enum ConcreteFunctionImplementation {
	Implemented(Box<ImplementedFunctionNode>), Native(Box<NativeFunctionNode>)
}

#[derive(Debug, Clone)]
pub struct NativeFunctionNode {
	annotation: Annotation
}

impl NativeFunctionNode {
	pub fn new(annotation: Annotation) -> Self {
		NativeFunctionNode {
			annotation
		}
	}
}

impl FunctionImplementationNode for NativeFunctionNode {
	fn get_concrete<'a>(&'a self) -> ConcreteFunctionImplementationRef<'a> {
		ConcreteFunctionImplementationRef::Native(&self)
	}

	fn into_concrete(self: Box<Self>) -> ConcreteFunctionImplementation {
		ConcreteFunctionImplementation::Native(self)
	}
	
	fn dyn_clone(&self) -> Box<dyn FunctionImplementationNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
pub struct ImplementedFunctionNode {
	annotation: Annotation,
	pub body: Box<StmtsNode>
}

impl ImplementedFunctionNode {
	pub fn new(annotation: Annotation, body: Box<StmtsNode>) -> Self {
		ImplementedFunctionNode {
			annotation, body
		}
	}
}

impl FunctionImplementationNode for ImplementedFunctionNode {
	fn get_concrete<'a>(&'a self) -> ConcreteFunctionImplementationRef<'a> {
		ConcreteFunctionImplementationRef::Implemented(&self)
	}

	fn into_concrete(self: Box<Self>) -> ConcreteFunctionImplementation {
		ConcreteFunctionImplementation::Implemented(self)
	}

	fn dyn_clone(&self) -> Box<dyn FunctionImplementationNode> {
		Box::new(self.clone())
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

impl Clone for ParameterNode {
	fn clone(&self) -> ParameterNode {
		ParameterNode {
			annotation: self.annotation.clone(),
			ident: self.ident.clone(),
			param_type: self.param_type.dyn_clone()
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

impl Clone for StmtsNode {
	fn clone(&self) -> StmtsNode {
		StmtsNode {
			annotation: self.annotation.clone(),
			stmts: self.stmts.iter().map(|el| (**el).dyn_clone()).collect()
		}
	}
}

pub trait StmtNode : Node {
	fn get_concrete<'a>(&'a self) -> ConcreteStmtRef<'a>;
	fn into_concrete(self: Box<Self>) -> ConcreteStmt;
	fn dyn_clone(&self) -> Box<dyn StmtNode>;
}

pub enum ConcreteStmtRef<'a> {
	Declaration(&'a VariableDeclarationNode), 
	Assignment(&'a AssignmentNode), 
	Expr(&'a ExprStmtNode), 
	If(&'a IfNode), 
	While(&'a WhileNode), 
	Block(&'a BlockNode), 
	Return(&'a ReturnNode)
}

pub enum ConcreteStmt {
	Declaration(Box<VariableDeclarationNode>), 
	Assignment(Box<AssignmentNode>), 
	Expr(Box<ExprStmtNode>), 
	If(Box<IfNode>), 
	While(Box<WhileNode>), 
	Block(Box<BlockNode>), 
	Return(Box<ReturnNode>)
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

impl Clone for VariableDeclarationNode {
	fn clone(&self) -> VariableDeclarationNode {
		VariableDeclarationNode {
			annotation: self.annotation.clone(),
			variable_type: self.variable_type.dyn_clone(),
			ident: self.ident.clone(),
			expr: self.expr.clone()
		}
	}
}

impl StmtNode for VariableDeclarationNode {
	fn get_concrete<'a>(&'a self) -> ConcreteStmtRef<'a> {
		ConcreteStmtRef::Declaration(&self)
	}

	fn into_concrete(self: Box<Self>) -> ConcreteStmt {
		ConcreteStmt::Declaration(self)
	}

	fn dyn_clone(&self) -> Box<dyn StmtNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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
	fn get_concrete<'a>(&'a self) -> ConcreteStmtRef<'a> {
		ConcreteStmtRef::Assignment(&self)
	}

	fn into_concrete(self: Box<Self>) -> ConcreteStmt {
		ConcreteStmt::Assignment(self)
	}

	fn dyn_clone(&self) -> Box<dyn StmtNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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
	fn get_concrete<'a>(&'a self) -> ConcreteStmtRef<'a> {
		ConcreteStmtRef::Expr(&self)
	}

	fn into_concrete(self: Box<Self>) -> ConcreteStmt {
		ConcreteStmt::Expr(self)
	}

	fn dyn_clone(&self) -> Box<dyn StmtNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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
	fn get_concrete<'a>(&'a self) -> ConcreteStmtRef<'a> {
		ConcreteStmtRef::If(&self)
	}

	fn into_concrete(self: Box<Self>) -> ConcreteStmt {
		ConcreteStmt::If(self)
	}

	fn dyn_clone(&self) -> Box<dyn StmtNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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
	fn get_concrete<'a>(&'a self) -> ConcreteStmtRef<'a> {
		ConcreteStmtRef::While(&self)
	}

	fn into_concrete(self: Box<Self>) -> ConcreteStmt {
		ConcreteStmt::While(self)
	}

	fn dyn_clone(&self) -> Box<dyn StmtNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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
	fn get_concrete<'a>(&'a self) -> ConcreteStmtRef<'a> {
		ConcreteStmtRef::Block(&self)
	}

	fn into_concrete(self: Box<Self>) -> ConcreteStmt {
		ConcreteStmt::Block(self)
	}

	fn dyn_clone(&self) -> Box<dyn StmtNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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
	fn get_concrete<'a>(&'a self) -> ConcreteStmtRef<'a> {
		ConcreteStmtRef::Return(&self)
	}

	fn into_concrete(self: Box<Self>) -> ConcreteStmt {
		ConcreteStmt::Return(self)
	}

	fn dyn_clone(&self) -> Box<dyn StmtNode> {
		Box::new(self.clone())
	}
}

pub trait TypeNode : Node {
	fn get_kind<'a>(&'a self) -> TypeKind<'a>;
	fn dyn_clone(&self) -> Box<dyn TypeNode>;
}

impl PartialEq for dyn TypeNode {
	fn eq(&self, other: &Self) -> bool {
		self.get_kind() == other.get_kind()
	}
}

#[derive(PartialEq)]
pub enum TypeKind<'a> {
	Array(&'a ArrTypeNode), Void(&'a VoidTypeNode)
}

#[derive(Debug)]
pub struct ArrTypeNode {
	annotation: Annotation,
	pub base_type: Box<dyn BaseTypeNode>,
	pub dims: u8
}

impl PartialEq for ArrTypeNode {
	fn eq(&self, other: &Self) -> bool {
		*self.base_type == *other.base_type && self.dims == other.dims
	}
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

impl Clone for ArrTypeNode {
	fn clone(&self) -> ArrTypeNode {
		ArrTypeNode {
			annotation: self.annotation.clone(),
			base_type: self.base_type.dyn_clone(),
			dims: self.dims
		}
	}
}

impl TypeNode for ArrTypeNode {
	fn get_kind<'a>(&'a self) -> TypeKind<'a> {
		TypeKind::Array(&self)
	}

	fn dyn_clone(&self) -> Box<dyn TypeNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
pub struct VoidTypeNode {
	annotation: Annotation
}

impl PartialEq for VoidTypeNode {
	fn eq(&self, other: &Self) -> bool {
		true
	}
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

	fn dyn_clone(&self) -> Box<dyn TypeNode> {
		Box::new(self.clone())
	}
}

pub type ExprNode = ExprNodeLvlOr;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

impl Clone for ExprNodeLvlCmp {
	fn clone(&self) -> ExprNodeLvlCmp {
		ExprNodeLvlCmp {
			annotation: self.annotation.clone(),
			head: self.head.clone(),
			tail: self.tail.iter().map(|el| (**el).dyn_clone()).collect()
		}
	}
}

pub trait CmpPartNode : Node {
	fn get_kind<'a>(&'a self) -> CmpPartKind<'a>;
	fn get_expr(&self) -> &ExprNodeLvlAdd;
	fn dyn_clone(&self) -> Box<dyn CmpPartNode>;
}

pub enum CmpPartKind<'a> {
	Eq(&'a CmpPartNodeEq), 
	Neq(&'a CmpPartNodeNeq), 
	Leq(&'a CmpPartNodeLeq), 
	Geq(&'a CmpPartNodeGeq), 
	Ls(&'a CmpPartNodeLs), 
	Gt(&'a CmpPartNodeGt)
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> {
		Box::new(self.clone())
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

impl Clone for ExprNodeLvlAdd {
	fn clone(&self) -> ExprNodeLvlAdd {
		ExprNodeLvlAdd {
			annotation: self.annotation.clone(),
			head: self.head.clone(),
			tail: self.tail.iter().map(|el| (**el).dyn_clone()).collect()
		}
	}
}

pub trait SumPartNode : Node {
	fn get_kind<'a>(&'a self) -> SumPartKind<'a>;
	fn get_expr(&self) -> &ExprNodeLvlMult;
	fn dyn_clone(&self) -> Box<dyn SumPartNode>;
}

pub enum SumPartKind<'a> {
	Add(&'a SumPartNodeAdd), Subtract(&'a SumPartNodeSub)
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn SumPartNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn SumPartNode> {
		Box::new(self.clone())
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

impl Clone for ExprNodeLvlMult {
	fn clone(&self) -> ExprNodeLvlMult {
		ExprNodeLvlMult {
			annotation: self.annotation.clone(),
			head: self.head.clone(),
			tail: self.tail.iter().map(|el| (**el).dyn_clone()).collect()
		}
	}
}

pub trait ProductPartNode : Node {
	fn get_kind<'a>(&'a self) -> ProductPartKind<'a>;
	fn get_expr(&self) -> &ExprNodeLvlIndex;
	fn dyn_clone(&self) -> Box<dyn ProductPartNode>;
}

pub enum ProductPartKind<'a> {
	Mult(&'a ProductPartNodeMult), Divide(&'a ProductPartNodeDivide)
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn ProductPartNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn ProductPartNode> {
		Box::new(self.clone())
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

impl Clone for ExprNodeLvlIndex {
	fn clone(&self) -> ExprNodeLvlIndex {
		ExprNodeLvlIndex {
			annotation: self.annotation.clone(),
			head: self.head.dyn_clone(),
			tail: self.tail.clone()
		}
	}
}

#[derive(Debug, Clone)]
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
	fn dyn_clone(&self) -> Box<dyn UnaryExprNode>;
}

pub enum UnaryExprKind<'a> {
	BracketExpr(&'a BracketExprNode), 
	Literal(&'a LiteralNode), 
	Variable(&'a VariableNode), 
	FunctionCall(&'a FunctionCallNode), 
	NewExpr(&'a NewExprNode)
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn UnaryExprNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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
	
	fn dyn_clone(&self) -> Box<dyn UnaryExprNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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
	
	fn dyn_clone(&self) -> Box<dyn UnaryExprNode> {
		Box::new(self.clone())
	}
}

#[derive(Debug, Clone)]
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

	fn dyn_clone(&self) -> Box<dyn UnaryExprNode> {
		Box::new(self.clone())
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

impl Clone for NewExprNode {
	fn clone(&self) -> NewExprNode {
		NewExprNode {
			annotation: self.annotation.clone(),
			base_type: self.base_type.dyn_clone(),
			dimensions: self.dimensions.clone()
		}
	}
}

impl UnaryExprNode for NewExprNode {
	fn get_kind<'a>(&'a self) -> UnaryExprKind<'a> {
		UnaryExprKind::NewExpr(&self)
	}

	fn dyn_clone(&self) -> Box<dyn UnaryExprNode> {
		Box::new(self.clone())
	}
}

pub trait BaseTypeNode : Node {
	fn get_kind<'a>(&'a self) -> BaseTypeKind<'a>;
	fn dyn_clone(&self) -> Box<dyn BaseTypeNode>;
}

impl PartialEq for dyn BaseTypeNode {
	fn eq(&self, other: &Self) -> bool {
		self.get_kind() == other.get_kind()
	}
}

#[derive(PartialEq)]
pub enum BaseTypeKind<'a> {
	Int(&'a IntTypeNode)
}

#[derive(Debug, Clone)]
pub struct IntTypeNode {
	annotation: Annotation
}

impl PartialEq for IntTypeNode {
	fn eq(&self, other: &Self) -> bool {
		true
	}
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
		BaseTypeKind::Int(self)
	}

	fn dyn_clone(&self) -> Box<dyn BaseTypeNode> {
		Box::new(self.clone())
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

			fn dynamic(&self) -> &(dyn Any + 'static) {
				self
			}

			fn dyn_clone_node(&self) -> Box<dyn Node> {
				Box::new(self.clone())
			}
		}
	};
}

impl_node!(FunctionNode);
impl_node!(NativeFunctionNode);
impl_node!(ImplementedFunctionNode);
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