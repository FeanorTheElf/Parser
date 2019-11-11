use super::super::lexer::tokens::{Identifier, Literal};
use super::super::lexer::position::TextPosition;
use super::super::lexer::error::CompileError;
use super::super::util::dyn_eq::DynEq;

use std::fmt::Debug;
use std::any::Any;

pub type AstVec<T> = Vec<Box<T>>;
type Annotation = TextPosition;

macro_rules! impl_concrete_node {
	($supnode:ident for $subnode:ident; $concreteref:ident; $concretemut:ident; $variant:ident) => {
				
		impl $supnode for $subnode {
			fn get_concrete<'a>(&'a self) -> $concreteref<'a> {
				$concreteref::$variant(&self)
			}

			fn get_mut_concrete<'a>(&'a mut self) -> $concretemut<'a> {
				$concretemut::$variant(self)
			}

			fn dyn_clone(&self) -> Box<dyn $supnode> {
				Box::new(self.clone())
			}
		}

	}
}

macro_rules! impl_get_generalized {
	($type:ident; $generalized:ident; $($variant:ident),*) => {
		impl<'a> $type<'a> {
			pub fn get_generalized(&self) -> &'a dyn $generalized {
				match self {
					$($type::$variant(ref x) => *x),*
				}
			}
		}
	};
}

macro_rules! impl_get_mut_generalized {
	($type:ident; $generalized:ident; $($variant:ident),*) => {
		impl<'a> $type<'a> {
			pub fn get_mut_generalized(self) -> &'a mut dyn $generalized {
				match self {
					$($type::$variant(x) => x),*
				}
			}
		}
	};
}

macro_rules! cmp_attributes {
	($fst:ident; $snd:ident; $attr:ident) => {
		&(($fst).$attr) == &(($snd).$attr)
	};
}

macro_rules! impl_partial_eq {
	($type:ident; ) => {
		impl PartialEq<$type> for $type {
			fn eq(&self, rhs: &$type) -> bool {
				true
			}
		}
	};
	($type:ident; $fst_attr:ident) => {
		impl PartialEq<$type> for $type {
			fn eq(&self, rhs: &$type) -> bool {
				cmp_attributes!(self; rhs; $fst_attr)
			}
		}
	};
	($type:ident; $fst_attr:ident, $($attr:ident),*) => {
		impl PartialEq<$type> for $type {
			fn eq(&self, rhs: &$type) -> bool {
				cmp_attributes!(self; rhs; $fst_attr) && $(cmp_attributes!(self; rhs; $attr))&&*
			}
		}
	};
	(dyn $type:ident) => {
		impl PartialEq<dyn $type> for dyn $type {
			fn eq(&self, rhs: &dyn $type) -> bool {
				(*self).dyn_eq(rhs.dynamic())
			}
		}
	}
}

pub trait Node : Debug + Any + DynEq {
	fn get_annotation(&self) -> &Annotation;
	fn get_annotation_mut(&mut self) -> &mut Annotation;
	fn dyn_clone_node(&self) -> Box<dyn Node>;
	fn dynamic(&self) -> &dyn Any;
}

type VisitorReturnType = Result<(), CompileError>;
type VisitorFunctionType<'a, 'b, T> = dyn 'b + FnMut(&'a T) -> VisitorReturnType;

pub trait Visitable<T: ?Sized> {
    fn iterate<'a, 'b>(&'a self, f: &'b mut VisitorFunctionType<'a, 'b, T>) -> VisitorReturnType;
}

type TransformResultType = ();
type TransformFunctionType<T> = dyn FnMut(Box<T>) -> Box<T>;

pub trait Transformable<T: ?Sized>
{
    fn transform(&mut self, f: &mut TransformFunctionType<T>) -> TransformResultType;
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

impl_partial_eq!(FunctionNode; annotation, ident, params, result, implementation);

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

pub trait FunctionImplementationNode : Node + Visitable<dyn UnaryExprNode> + 
	Transformable<dyn StmtNode> + Transformable<dyn UnaryExprNode>  
{
	fn get_concrete<'a>(&'a self) -> ConcreteFunctionImplementationRef<'a>;
	fn get_mut_concrete<'a>(&'a mut self) -> ConcreteFunctionImplementationMut<'a>;
	fn dyn_clone(&self) -> Box<dyn FunctionImplementationNode>;
}

impl_partial_eq!(dyn FunctionImplementationNode);

pub enum ConcreteFunctionImplementationRef<'a> {
	Implemented(&'a ImplementedFunctionNode), Native(&'a NativeFunctionNode)
}

impl_get_generalized!(ConcreteFunctionImplementationRef; FunctionImplementationNode; Implemented, Native);

pub enum ConcreteFunctionImplementationMut<'a> {
	Implemented(&'a mut ImplementedFunctionNode), Native(&'a mut NativeFunctionNode)
}

impl_get_mut_generalized!(ConcreteFunctionImplementationMut; FunctionImplementationNode; Implemented, Native);

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

impl_concrete_node!(FunctionImplementationNode for NativeFunctionNode; ConcreteFunctionImplementationRef; ConcreteFunctionImplementationMut; Native);

impl_partial_eq!(NativeFunctionNode; );

#[derive(Debug, Clone)]
pub struct ImplementedFunctionNode {
	annotation: Annotation,
	pub body: Box<BlockNode>
}

impl ImplementedFunctionNode {
	pub fn new(annotation: Annotation, body: Box<BlockNode>) -> Self {
		ImplementedFunctionNode {
			annotation, body
		}
	}
}

impl_concrete_node!(FunctionImplementationNode for ImplementedFunctionNode; ConcreteFunctionImplementationRef; ConcreteFunctionImplementationMut; Implemented);

impl_partial_eq!(ImplementedFunctionNode; body);

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

impl_partial_eq!(ParameterNode; ident, param_type);

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
pub struct BlockNode {
	annotation: Annotation,
	pub stmts: AstVec<dyn StmtNode>
}

impl BlockNode {
	pub fn new(annotation: Annotation, stmts: AstVec<dyn StmtNode>) -> Self {
		BlockNode {
			annotation, stmts
		}
	}
}

impl_partial_eq!(BlockNode; stmts);

impl Clone for BlockNode {
	fn clone(&self) -> BlockNode {
		BlockNode {
			annotation: self.annotation.clone(),
			stmts: self.stmts.iter().map(|el| (**el).dyn_clone()).collect()
		}
	}
}

pub trait StmtNode : Node + Visitable<dyn UnaryExprNode> + Transformable<dyn UnaryExprNode> {
	fn get_concrete<'a>(&'a self) -> ConcreteStmtRef<'a>;
	fn get_mut_concrete<'a>(&'a mut self) -> ConcreteStmtMut<'a>;
	fn dyn_clone(&self) -> Box<dyn StmtNode>;
}

impl_partial_eq!(dyn StmtNode);

pub enum ConcreteStmtRef<'a> {
	Declaration(&'a VariableDeclarationNode), 
	Assignment(&'a AssignmentNode), 
	Expr(&'a ExprStmtNode), 
	If(&'a IfNode), 
	While(&'a WhileNode), 
	Block(&'a BlockNode), 
	Return(&'a ReturnNode)
}

impl_get_generalized!(ConcreteStmtRef; StmtNode; Declaration, Assignment, Expr, If, While, Block, Return);

pub enum ConcreteStmtMut<'a> {
	Declaration(&'a mut VariableDeclarationNode), 
	Assignment(&'a mut AssignmentNode), 
	Expr(&'a mut ExprStmtNode), 
	If(&'a mut IfNode), 
	While(&'a mut WhileNode), 
	Block(&'a mut BlockNode), 
	Return(&'a mut ReturnNode)
}

impl_get_mut_generalized!(ConcreteStmtMut; StmtNode; Declaration, Assignment, Expr, If, While, Block, Return);

impl_concrete_node!(StmtNode for BlockNode; ConcreteStmtRef; ConcreteStmtMut; Block);

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

impl_concrete_node!(StmtNode for VariableDeclarationNode; ConcreteStmtRef; ConcreteStmtMut; Declaration);

impl_partial_eq!(VariableDeclarationNode; variable_type, ident, expr);

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

impl_concrete_node!(StmtNode for AssignmentNode; ConcreteStmtRef; ConcreteStmtMut; Assignment);

impl_partial_eq!(AssignmentNode; assignee, expr);

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

impl_concrete_node!(StmtNode for ExprStmtNode; ConcreteStmtRef; ConcreteStmtMut; Expr);

impl_partial_eq!(ExprStmtNode; expr);

#[derive(Debug, Clone)]
pub struct IfNode {
	annotation: Annotation,
	pub condition: Box<ExprNode>,
	pub block: Box<BlockNode>
}

impl IfNode {
	pub fn new(annotation: Annotation, condition: Box<ExprNode>, block: Box<BlockNode>) -> Self {
		IfNode {
			annotation, condition, block
		}
	}
}

impl_concrete_node!(StmtNode for IfNode; ConcreteStmtRef; ConcreteStmtMut; If);

impl_partial_eq!(IfNode; condition, block);

#[derive(Debug, Clone)]
pub struct WhileNode {
	annotation: Annotation,
	pub condition: Box<ExprNode>,
	pub block: Box<BlockNode>
}

impl WhileNode {
	pub fn new(annotation: Annotation, condition: Box<ExprNode>, block: Box<BlockNode>) -> Self {
		WhileNode {
			annotation, condition, block
		}
	}
}

impl_concrete_node!(StmtNode for WhileNode; ConcreteStmtRef; ConcreteStmtMut; While);

impl_partial_eq!(WhileNode; condition, block);

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

impl_concrete_node!(StmtNode for ReturnNode; ConcreteStmtRef; ConcreteStmtMut; Return);

impl_partial_eq!(ReturnNode; expr);

pub trait TypeNode : Node {
	fn get_concrete<'a>(&'a self) -> ConcreteTypeRef<'a>;
	fn get_mut_concrete<'a>(&'a mut self) -> ConcreteTypeMut<'a>;
	fn dyn_clone(&self) -> Box<dyn TypeNode>;
}

impl_partial_eq!(dyn TypeNode);

pub enum ConcreteTypeRef<'a> {
	Array(&'a ArrTypeNode), Void(&'a VoidTypeNode)
}

impl_get_generalized!(ConcreteTypeRef; TypeNode; Array, Void);

pub enum ConcreteTypeMut<'a> {
	Array(&'a mut ArrTypeNode), Void(&'a mut VoidTypeNode)
}

impl_get_mut_generalized!(ConcreteTypeMut; TypeNode; Array, Void);

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

impl_partial_eq!(ArrTypeNode; base_type, dims);

impl Clone for ArrTypeNode {
	fn clone(&self) -> ArrTypeNode {
		ArrTypeNode {
			annotation: self.annotation.clone(),
			base_type: self.base_type.dyn_clone(),
			dims: self.dims
		}
	}
}

impl_concrete_node!(TypeNode for ArrTypeNode; ConcreteTypeRef; ConcreteTypeMut; Array);

#[derive(Debug, Clone)]
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

impl_concrete_node!(TypeNode for VoidTypeNode; ConcreteTypeRef; ConcreteTypeMut; Void);

impl_partial_eq!(VoidTypeNode;);

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

impl_partial_eq!(ExprNodeLvlOr; head, tail);

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

impl_partial_eq!(OrPartNode; expr);

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

impl_partial_eq!(ExprNodeLvlAnd; head, tail);

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

impl_partial_eq!(AndPartNode; expr);

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

impl_partial_eq!(ExprNodeLvlCmp; head, tail);

impl Clone for ExprNodeLvlCmp {
	fn clone(&self) -> ExprNodeLvlCmp {
		ExprNodeLvlCmp {
			annotation: self.annotation.clone(),
			head: self.head.clone(),
			tail: self.tail.iter().map(|el| (**el).dyn_clone()).collect()
		}
	}
}

pub trait CmpPartNode : Node + Transformable<dyn UnaryExprNode> {
	fn get_kind<'a>(&'a self) -> CmpPartKind<'a>;
	fn get_expr(&self) -> &ExprNodeLvlAdd;
	fn dyn_clone(&self) -> Box<dyn CmpPartNode>;
}

impl_partial_eq!(dyn CmpPartNode);

pub enum CmpPartKind<'a> {
	Eq(&'a CmpPartNodeEq), 
	Neq(&'a CmpPartNodeNeq), 
	Leq(&'a CmpPartNodeLeq), 
	Geq(&'a CmpPartNodeGeq), 
	Ls(&'a CmpPartNodeLs), 
	Gt(&'a CmpPartNodeGt)
}

impl_get_generalized!(CmpPartKind; CmpPartNode; Eq, Neq, Leq, Geq, Ls, Gt);

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

impl_partial_eq!(CmpPartNodeEq; expr);

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

impl_partial_eq!(CmpPartNodeNeq; expr);

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

impl_partial_eq!(CmpPartNodeLeq; expr);

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

impl_partial_eq!(CmpPartNodeGeq; expr);

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

impl_partial_eq!(CmpPartNodeLs; expr);

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

impl_partial_eq!(CmpPartNodeGt; expr);

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

impl_partial_eq!(ExprNodeLvlAdd; head, tail);

impl Clone for ExprNodeLvlAdd {
	fn clone(&self) -> ExprNodeLvlAdd {
		ExprNodeLvlAdd {
			annotation: self.annotation.clone(),
			head: self.head.clone(),
			tail: self.tail.iter().map(|el| (**el).dyn_clone()).collect()
		}
	}
}

pub trait SumPartNode : Node + Transformable<dyn UnaryExprNode> {
	fn get_kind<'a>(&'a self) -> SumPartKind<'a>;
	fn get_expr(&self) -> &ExprNodeLvlMult;
	fn dyn_clone(&self) -> Box<dyn SumPartNode>;
}

impl_partial_eq!(dyn SumPartNode);

pub enum SumPartKind<'a> {
	Add(&'a SumPartNodeAdd), Subtract(&'a SumPartNodeSub)
}

impl_get_generalized!(SumPartKind; SumPartNode; Add, Subtract);

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

impl_partial_eq!(SumPartNodeAdd; expr);

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

impl_partial_eq!(SumPartNodeSub; expr);

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

impl_partial_eq!(ExprNodeLvlMult; head, tail);

impl Clone for ExprNodeLvlMult {
	fn clone(&self) -> ExprNodeLvlMult {
		ExprNodeLvlMult {
			annotation: self.annotation.clone(),
			head: self.head.clone(),
			tail: self.tail.iter().map(|el| (**el).dyn_clone()).collect()
		}
	}
}

pub trait ProductPartNode : Node + Transformable<dyn UnaryExprNode> {
	fn get_kind<'a>(&'a self) -> ProductPartKind<'a>;
	fn get_expr(&self) -> &ExprNodeLvlIndex;
	fn dyn_clone(&self) -> Box<dyn ProductPartNode>;
}

impl_partial_eq!(dyn ProductPartNode);

pub enum ProductPartKind<'a> {
	Mult(&'a ProductPartNodeMult), Divide(&'a ProductPartNodeDivide)
}

impl_get_generalized!(ProductPartKind; ProductPartNode; Mult, Divide);

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

impl_partial_eq!(ProductPartNodeMult; expr);

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

impl_partial_eq!(ProductPartNodeDivide; expr);

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

impl_partial_eq!(ExprNodeLvlIndex; head, tail);

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

impl_partial_eq!(IndexPartNode; expr);

pub trait UnaryExprNode : Node {
	fn get_concrete<'a>(&'a self) -> ConcreteUnaryExprRef<'a>;
	fn get_mut_concrete<'a>(&'a mut self) -> ConcreteUnaryExprMut<'a>;
	fn dyn_clone(&self) -> Box<dyn UnaryExprNode>;
}

impl_partial_eq!(dyn UnaryExprNode);

pub enum ConcreteUnaryExprRef<'a> {
	BracketExpr(&'a BracketExprNode), 
	Literal(&'a LiteralNode), 
	Variable(&'a VariableNode), 
	FunctionCall(&'a FunctionCallNode), 
	NewExpr(&'a NewExprNode)
}

impl_get_generalized!(ConcreteUnaryExprRef; UnaryExprNode; BracketExpr, Literal, Variable, FunctionCall, NewExpr);

pub enum ConcreteUnaryExprMut<'a> {
	BracketExpr(&'a mut BracketExprNode), 
	Literal(&'a mut LiteralNode), 
	Variable(&'a mut VariableNode), 
	FunctionCall(&'a mut FunctionCallNode), 
	NewExpr(&'a mut NewExprNode)
}

impl_get_mut_generalized!(ConcreteUnaryExprMut; UnaryExprNode; BracketExpr, Literal, Variable, FunctionCall, NewExpr);

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

impl_concrete_node!(UnaryExprNode for BracketExprNode; ConcreteUnaryExprRef; ConcreteUnaryExprMut; BracketExpr);

impl_partial_eq!(BracketExprNode; expr);

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

impl_concrete_node!(UnaryExprNode for LiteralNode; ConcreteUnaryExprRef; ConcreteUnaryExprMut; Literal);

impl_partial_eq!(LiteralNode; literal);

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

impl_concrete_node!(UnaryExprNode for VariableNode; ConcreteUnaryExprRef; ConcreteUnaryExprMut; Variable);

impl_partial_eq!(VariableNode; identifier);

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

impl_concrete_node!(UnaryExprNode for FunctionCallNode; ConcreteUnaryExprRef; ConcreteUnaryExprMut; FunctionCall);

impl_partial_eq!(FunctionCallNode; function, params);

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

impl_concrete_node!(UnaryExprNode for NewExprNode; ConcreteUnaryExprRef; ConcreteUnaryExprMut; NewExpr);

impl_partial_eq!(NewExprNode; base_type, dimensions);

impl Clone for NewExprNode {
	fn clone(&self) -> NewExprNode {
		NewExprNode {
			annotation: self.annotation.clone(),
			base_type: self.base_type.dyn_clone(),
			dimensions: self.dimensions.clone()
		}
	}
}

pub trait BaseTypeNode : Node {
	fn get_concrete<'a>(&'a self) -> ConcreteBaseTypeRef<'a>;
	fn get_mut_concrete<'a>(&'a mut self) -> ConcreteBaseTypeMut<'a>;
	fn dyn_clone(&self) -> Box<dyn BaseTypeNode>;
}

impl_partial_eq!(dyn BaseTypeNode);

pub enum ConcreteBaseTypeRef<'a> {
	Int(&'a IntTypeNode)
}

impl_get_generalized!(ConcreteBaseTypeRef; BaseTypeNode; Int);

pub enum ConcreteBaseTypeMut<'a> {
	Int(&'a mut IntTypeNode)
}

impl_get_mut_generalized!(ConcreteBaseTypeMut; BaseTypeNode; Int);

#[derive(Debug, Clone)]
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

impl_concrete_node!(BaseTypeNode for IntTypeNode; ConcreteBaseTypeRef; ConcreteBaseTypeMut; Int);

impl_partial_eq!(IntTypeNode;);

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
impl_node!(BlockNode);
impl_node!(VariableDeclarationNode);
impl_node!(AssignmentNode);
impl_node!(ExprStmtNode);
impl_node!(IfNode);
impl_node!(WhileNode);
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


impl Visitable<dyn UnaryExprNode> for FunctionNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        self.implementation.iterate(f)
    }
}

impl Visitable<dyn UnaryExprNode> for NativeFunctionNode {
    fn iterate<'a, 'b>(&'a self, f: &mut VisitorFunctionType<'a, 'b, dyn UnaryExprNode>) -> VisitorReturnType
    {
        Ok(())
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


impl Transformable<dyn StmtNode> for FunctionNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn StmtNode>) -> TransformResultType
    {
        self.implementation.transform(f);
    }
}

impl Transformable<dyn StmtNode> for ImplementedFunctionNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn StmtNode>) -> TransformResultType
    {
        self.body.transform(f);
    }
}

impl Transformable<dyn StmtNode> for NativeFunctionNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn StmtNode>) -> TransformResultType
    {
    }
}

impl Transformable<dyn StmtNode> for BlockNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn StmtNode>) -> TransformResultType
    {
        self.stmts = self.stmts.drain(..).map(f).collect();
    }
}

impl Transformable<dyn StmtNode> for IfNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn StmtNode>) -> TransformResultType
    {
        self.block.transform(f)
    }
}

impl Transformable<dyn StmtNode> for WhileNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn StmtNode>) -> TransformResultType
    {
        self.block.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for FunctionNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.implementation.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for ImplementedFunctionNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.body.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for NativeFunctionNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
    }
}

impl Transformable<dyn UnaryExprNode> for BlockNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.stmts.iter_mut().for_each(|stmt| stmt.transform(f));
    }
}

impl Transformable<dyn UnaryExprNode> for AssignmentNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.assignee.transform(f);
        self.expr.transform(f);
    }
}

impl Transformable<dyn UnaryExprNode> for VariableDeclarationNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f);
    }
}

impl Transformable<dyn UnaryExprNode> for ExprStmtNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f);
    }
}

impl Transformable<dyn UnaryExprNode> for IfNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.condition.transform(f);
		self.block.transform(f);
    }
}

impl Transformable<dyn UnaryExprNode> for WhileNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.condition.transform(f);
		self.block.transform(f);
    }
}

impl Transformable<dyn UnaryExprNode> for ReturnNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
		self.expr.transform(f);
    }
}

impl Transformable<dyn UnaryExprNode> for ExprNodeLvlOr {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.head.transform(f);
		self.tail.iter_mut().for_each(|part| part.transform(f));
    }
}

impl Transformable<dyn UnaryExprNode> for OrPartNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for ExprNodeLvlAnd {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.head.transform(f);
		self.tail.iter_mut().for_each(|part| part.transform(f));
    }
}

impl Transformable<dyn UnaryExprNode> for AndPartNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for ExprNodeLvlCmp {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.head.transform(f);
		self.tail.iter_mut().for_each(|part| part.transform(f));
    }
}

impl Transformable<dyn UnaryExprNode> for CmpPartNodeEq {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for CmpPartNodeNeq {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for CmpPartNodeLeq {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for CmpPartNodeGeq {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for CmpPartNodeLs {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for CmpPartNodeGt {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for ExprNodeLvlAdd {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.head.transform(f);
		self.tail.iter_mut().for_each(|part| part.transform(f));
    }
}

impl Transformable<dyn UnaryExprNode> for SumPartNodeSub {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for SumPartNodeAdd {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for ExprNodeLvlMult {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.head.transform(f);
		self.tail.iter_mut().for_each(|part| part.transform(f));
    }
}

impl Transformable<dyn UnaryExprNode> for ProductPartNodeMult {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for ProductPartNodeDivide {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

impl Transformable<dyn UnaryExprNode> for ExprNodeLvlIndex {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        take_mut::take(&mut self.head, |expr| f(expr));
		self.tail.iter_mut().for_each(|part| part.transform(f));
    }
}

impl Transformable<dyn UnaryExprNode> for IndexPartNode {
    fn transform(&mut self, f: &mut TransformFunctionType<dyn UnaryExprNode>) -> TransformResultType
    {
        self.expr.transform(f)
    }
}

#[cfg(test)]
use super::super::parser::Parse;
#[cfg(test)]
use super::super::lexer::lexer::lex;
#[cfg(test)]
use test::Bencher;

#[cfg(test)]
fn rek_transform(mut stmt: Box<dyn StmtNode>) -> Box<dyn StmtNode> {
    match stmt.get_mut_concrete() {
        ConcreteStmtMut::Block(stmts) => stmts.transform(&mut rek_transform),
        ConcreteStmtMut::If(stmt) => stmt.block.transform(&mut rek_transform),
        ConcreteStmtMut::While(stmt) => stmt.block.transform(&mut rek_transform),
        _ => ()
    };

    let global_function_call = ExprNode::parse(&mut lex("global()")).unwrap();
    let global_function_call_stmt = Box::new(ExprStmtNode::new(TextPosition::create(0, 0), global_function_call));

    let stmts = vec![stmt, global_function_call_stmt.clone()];
    let result: Box<dyn StmtNode> = Box::new(BlockNode::new(TextPosition::create(0, 0), stmts));
	return result;
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
    transformed_function.transform(&mut rek_transform);
    assert_eq!(expected_function, transformed_function);
}

#[bench]
fn bench_visitor(bencher: &mut Bencher) {
    let ast = *FunctionNode::parse(&mut lex("fn foo(a: int,): int[] {
        let result: int[] = new int[a];
        let index: int = identity(a);
        while (index > 0) {
            index = decrement(index);
            result[index] = calculate(index);
        }
        index = identity(a);
        while (index > 0) {
            index = decrement(index);
            result[index] = result[index] + bar(result, index)[index];
        }
        return postprocess(result);
    }")).unwrap();
    let mut elements: Vec<&FunctionCallNode> = vec![];
    bencher.iter(|| {
        elements.clear();
        ast.iterate(&mut |expr: &dyn UnaryExprNode| {
            if let Some(call) = expr.dynamic().downcast_ref::<FunctionCallNode>() {
                elements.push(call);
            }
            return Ok(());
        });
        assert_eq!(7, elements.len());
    })
}