use super::super::lexer::tokens::{ Identifier, Literal };
use super::super::lexer::error::CompileError;

use super::ast::*;
use super::visitor::{ Visitable, Transformable, Visitor, Transformer };
use super::obj_type::*;

use std::any::Any;

pub type ExprNode = ExprNodeLvlOr;

#[derive(Debug, Clone)]
pub struct ExprNodeLvlOr 
{
	annotation: Annotation,
	pub head: Box<ExprNodeLvlAnd>,
	pub tail: AstVec<OrPartNode>
}

impl ExprNodeLvlOr 
{
    pub fn new(annotation: Annotation, head: Box<ExprNodeLvlAnd>, tail: AstVec<OrPartNode>) -> Self 
    {
		ExprNodeLvlOr {
			annotation, head, tail
		}
	}
}

impl_visitable!(ExprNodeLvlOr; head, vec tail);
impl_transformable!(ExprNodeLvlOr; head, vec tail);
impl_partial_eq!(ExprNodeLvlOr; head, tail);

#[derive(Debug, Clone)]
pub struct OrPartNode {
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlAnd>
}

impl OrPartNode 
{
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAnd>) -> Self 
	{
		OrPartNode {
			annotation, expr
		}
	}
}

impl_transformable!(OrPartNode; expr);
impl_visitable!(OrPartNode; expr);
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

impl_visitable!(ExprNodeLvlAnd; head, vec tail);
impl_transformable!(ExprNodeLvlAnd; head, vec tail);
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

impl_visitable!(AndPartNode; expr);
impl_transformable!(AndPartNode; expr);
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

impl_visitable!(ExprNodeLvlCmp; head, vec tail);
impl_transformable!(ExprNodeLvlCmp; head, vec tail);
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

pub trait CmpPartNode : Node
{
	fn get_expr(&self) -> &ExprNodeLvlAdd;
	fn dyn_clone(&self) -> Box<dyn CmpPartNode>;
}

impl_partial_eq!(dyn CmpPartNode);

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

impl CmpPartNode for CmpPartNodeEq 
{
	fn get_expr(&self) -> &ExprNodeLvlAdd
	{
		&*self.expr
	}

	fn dyn_clone(&self) -> Box<dyn CmpPartNode>
	{
		Box::new(self.clone())
	}
}

impl_visitable!(CmpPartNodeEq; expr);
impl_transformable!(CmpPartNodeEq; expr);
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

impl CmpPartNode for CmpPartNodeNeq 
{
	fn get_expr(&self) -> &ExprNodeLvlAdd 
	{
		&*self.expr
	}

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> 
	{
		Box::new(self.clone())
	}
}

impl_visitable!(CmpPartNodeNeq; expr);
impl_transformable!(CmpPartNodeNeq; expr);
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

impl CmpPartNode for CmpPartNodeLeq 
{
	fn get_expr(&self) -> &ExprNodeLvlAdd 
	{
		&*self.expr
	}

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> {
		Box::new(self.clone())
	}
}

impl_visitable!(CmpPartNodeLeq; expr);
impl_transformable!(CmpPartNodeLeq; expr);
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

impl CmpPartNode for CmpPartNodeGeq 
{
	fn get_expr(&self) -> &ExprNodeLvlAdd 
	{
		&*self.expr
	}

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> 
	{
		Box::new(self.clone())
	}
}

impl_visitable!(CmpPartNodeGeq; expr);
impl_transformable!(CmpPartNodeGeq; expr);
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

impl CmpPartNode for CmpPartNodeLs 
{
	fn get_expr(&self) -> &ExprNodeLvlAdd 
	{
		&*self.expr
	}

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> 
	{
		Box::new(self.clone())
	}
}

impl_visitable!(CmpPartNodeLs; expr);
impl_transformable!(CmpPartNodeLs; expr);
impl_partial_eq!(CmpPartNodeLs; expr);

#[derive(Debug, Clone)]
pub struct CmpPartNodeGt 
{
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlAdd>
}

impl CmpPartNodeGt
 {
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlAdd>) -> Self 
	{
		CmpPartNodeGt {
			annotation, expr
		}
	}
}

impl CmpPartNode for CmpPartNodeGt 
{
	fn get_expr(&self) -> &ExprNodeLvlAdd 
	{
		&*self.expr
	}

	fn dyn_clone(&self) -> Box<dyn CmpPartNode> 
	{
		Box::new(self.clone())
	}
}

impl_visitable!(CmpPartNodeGt; expr);
impl_transformable!(CmpPartNodeGt; expr);
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

impl_visitable!(ExprNodeLvlAdd; head, vec tail);
impl_transformable!(ExprNodeLvlAdd; head, vec tail);
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

pub trait SumPartNode : Node
{
	fn get_expr(&self) -> &ExprNodeLvlMult;
	fn dyn_clone(&self) -> Box<dyn SumPartNode>;
}

impl_partial_eq!(dyn SumPartNode);

#[derive(Debug, Clone)]
pub struct SumPartNodeAdd 
{
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlMult>
}

impl SumPartNodeAdd 
{
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlMult>) -> Self 
	{
		SumPartNodeAdd {
			annotation, expr
		}
	}
}

impl SumPartNode for SumPartNodeAdd 
{
	fn get_expr(&self) -> &ExprNodeLvlMult 
	{
		&*self.expr
	}

	fn dyn_clone(&self) -> Box<dyn SumPartNode> 
	{
		Box::new(self.clone())
	}
}

impl_visitable!(SumPartNodeAdd; expr);
impl_transformable!(SumPartNodeAdd; expr);
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

impl SumPartNode for SumPartNodeSub 
{
	fn get_expr(&self) -> &ExprNodeLvlMult 
	{
		&*self.expr
	}

	fn dyn_clone(&self) -> Box<dyn SumPartNode> 
	{
		Box::new(self.clone())
	}
}

impl_visitable!(SumPartNodeSub; expr);
impl_transformable!(SumPartNodeSub; expr);
impl_partial_eq!(SumPartNodeSub; expr);

#[derive(Debug)]
pub struct ExprNodeLvlMult 
{
	annotation: Annotation,
	pub head: Box<ExprNodeLvlIndex>,
	pub tail: AstVec<dyn ProductPartNode>
}

impl ExprNodeLvlMult 
{
	pub fn new(annotation: Annotation, head: Box<ExprNodeLvlIndex>, tail: AstVec<dyn ProductPartNode>) -> Self 
	{
		ExprNodeLvlMult {
			annotation, head, tail
		}
	}
}

impl_visitable!(ExprNodeLvlMult; head, vec tail);
impl_transformable!(ExprNodeLvlMult; head, vec tail);
impl_partial_eq!(ExprNodeLvlMult; head, tail);

impl Clone for ExprNodeLvlMult 
{
	fn clone(&self) -> ExprNodeLvlMult 
	{
		ExprNodeLvlMult {
			annotation: self.annotation.clone(),
			head: self.head.clone(),
			tail: self.tail.iter().map(|el| (**el).dyn_clone()).collect()
		}
	}
}

pub trait ProductPartNode : Node + Visitable
{
	fn get_expr(&self) -> &ExprNodeLvlIndex;
	fn dyn_clone(&self) -> Box<dyn ProductPartNode>;
}

impl_partial_eq!(dyn ProductPartNode);

#[derive(Debug, Clone)]
pub struct ProductPartNodeMult 
{
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlIndex>
}

impl ProductPartNodeMult 
{
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlIndex>) -> Self 
	{
		ProductPartNodeMult {
			annotation, expr
		}
	}
}

impl ProductPartNode for ProductPartNodeMult 
{
	fn get_expr(&self) -> &ExprNodeLvlIndex 
	{
		&*self.expr
	}

	fn dyn_clone(&self) -> Box<dyn ProductPartNode> 
	{
		Box::new(self.clone())
	}
}

impl_transformable!(ProductPartNodeMult; expr);
impl_visitable!(ProductPartNodeMult; expr);
impl_partial_eq!(ProductPartNodeMult; expr);

#[derive(Debug, Clone)]
pub struct ProductPartNodeDivide 
{
	annotation: Annotation,
	pub expr: Box<ExprNodeLvlIndex>
}

impl ProductPartNodeDivide 
{
	pub fn new(annotation: Annotation, expr: Box<ExprNodeLvlIndex>) -> Self 
	{
		ProductPartNodeDivide {
			annotation, expr
		}
	}
}

impl ProductPartNode for ProductPartNodeDivide 
{
	fn get_expr(&self) -> &ExprNodeLvlIndex 
	{
		&*self.expr
	}

	fn dyn_clone(&self) -> Box<dyn ProductPartNode> 
	{
		Box::new(self.clone())
	}
}

impl_transformable!(ProductPartNodeDivide; expr);
impl_visitable!(ProductPartNodeDivide; expr);
impl_partial_eq!(ProductPartNodeDivide; expr);

#[derive(Debug)]
pub struct ExprNodeLvlIndex {
	annotation: Annotation,
	pub head: Box<dyn UnaryExprNode>,
	pub tail: AstVec<IndexPartNode>
}

impl ExprNodeLvlIndex 
{
	pub fn new(annotation: Annotation, head: Box<dyn UnaryExprNode>, tail: AstVec<IndexPartNode>) -> Self 
	{
		ExprNodeLvlIndex {
			annotation, head, tail
		}
	}
}

impl Transformable for ExprNodeLvlIndex
{
	fn transform(&mut self, transformer: &mut dyn Transformer)
	{
		transformer.before(self);
		take_mut::take(&mut self.head, |node| transformer.transform_expr(node));
		self.tail.iter_mut().for_each(|node| node.transform(transformer));
		transformer.after(self);
	}
}

impl_visitable!(ExprNodeLvlIndex; head, vec tail);
impl_partial_eq!(ExprNodeLvlIndex; head, tail);

impl Clone for ExprNodeLvlIndex 
{
	fn clone(&self) -> ExprNodeLvlIndex 
	{
		ExprNodeLvlIndex {
			annotation: self.annotation.clone(),
			head: self.head.dyn_clone(),
			tail: self.tail.clone()
		}
	}
}

#[derive(Debug, Clone)]
pub struct IndexPartNode 
{
	annotation: Annotation,
	pub expr: Box<ExprNode>
}

impl IndexPartNode 
{
	pub fn new(annotation: Annotation, expr: Box<ExprNode>) -> Self 
	{
		IndexPartNode {
			annotation, expr
		}
	}
}

impl_transformable!(IndexPartNode; expr);
impl_visitable!(IndexPartNode; expr);
impl_partial_eq!(IndexPartNode; expr);

#[derive(Debug, Clone)]
pub struct BracketExprNode 
{
	annotation: Annotation,
	pub expr: Box<ExprNode>
}

impl BracketExprNode 
{
	pub fn new(annotation: Annotation, expr: Box<ExprNode>) -> Self 
	{
		BracketExprNode {
			annotation, expr
		}
	}
}

impl_subnode!(UnaryExprNode for BracketExprNode);

impl_transformable!(BracketExprNode; expr);
impl_visitable!(BracketExprNode; expr);
impl_partial_eq!(BracketExprNode; expr);

#[derive(Debug, Clone)]
pub struct LiteralNode 
{
	annotation: Annotation,
	pub literal: Literal
}

impl LiteralNode 
{
	pub fn new(annotation: Annotation, literal: Literal) -> Self 
	{
		LiteralNode {
			annotation, literal
		}
	}
}

impl_subnode!(UnaryExprNode for LiteralNode);

impl_transformable!(LiteralNode;);
impl_visitable!(LiteralNode;);
impl_partial_eq!(LiteralNode; literal);

#[derive(Debug, Clone)]
pub struct VariableNode 
{
	annotation: Annotation,
	pub identifier: Identifier
}

impl VariableNode 
{
	pub fn new(annotation: Annotation, identifier: Identifier) -> Self 
	{
		VariableNode {
			annotation, identifier
		}
	}
}

impl_subnode!(UnaryExprNode for VariableNode);

impl_transformable!(VariableNode;);
impl_visitable!(VariableNode;);
impl_partial_eq!(VariableNode; identifier);

#[derive(Debug, Clone)]
pub struct FunctionCallNode 
{
	annotation: Annotation,
	pub function: Identifier,
	pub params: AstVec<ExprNode>
}

impl FunctionCallNode 
{
	pub fn new(annotation: Annotation, function: Identifier, params: AstVec<ExprNode>) -> Self 
	{
		FunctionCallNode {
			annotation, function, params
		}
	}
}

impl_subnode!(UnaryExprNode for FunctionCallNode);

impl_transformable!(FunctionCallNode; vec params);
impl_visitable!(FunctionCallNode; vec params);
impl_partial_eq!(FunctionCallNode; function, params);

#[derive(Debug)]
pub struct NewExprNode 
{
	annotation: Annotation,
	pub base_type: Box<dyn BaseTypeNode>,
	pub dimensions: AstVec<IndexPartNode>
}

impl NewExprNode 
{
	pub fn new(annotation: Annotation, base_type: Box<dyn BaseTypeNode>, dimensions: AstVec<IndexPartNode>) -> Self 
	{
		NewExprNode {
			annotation, base_type, dimensions
		}
	}
}

impl_subnode!(UnaryExprNode for NewExprNode);

impl_transformable!(NewExprNode; base_type, vec dimensions);
impl_visitable!(NewExprNode; base_type, vec dimensions);
impl_partial_eq!(NewExprNode; base_type, dimensions);

impl Clone for NewExprNode 
{
	fn clone(&self) -> NewExprNode 
	{
		NewExprNode {
			annotation: self.annotation.clone(),
			base_type: self.base_type.dyn_clone(),
			dimensions: self.dimensions.clone()
		}
	}
}

pub trait BaseTypeNode : Node + TypeDefinition
{
	fn dyn_clone(&self) -> Box<dyn BaseTypeNode>;
}

impl_partial_eq!(dyn BaseTypeNode);

#[derive(Debug, Clone)]
pub struct IntTypeNode {
	annotation: Annotation
}

impl IntTypeNode 
{
	pub fn new(annotation: Annotation) -> Self 
	{
		IntTypeNode {
			annotation
		}
	}
}

impl_subnode!(BaseTypeNode for IntTypeNode);

impl_visitable!(IntTypeNode;);
impl_transformable!(IntTypeNode;);
impl_partial_eq!(IntTypeNode;);

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

impl TypeDefinition for IntTypeNode
{
	fn calc_type(&self) -> Result<Option<Type>, CompileError>
	{
		Ok(Some(Type::Primitive(PrimitiveType::Int)))
	}
}