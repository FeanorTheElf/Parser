use super::super::lexer::tokens::{ Identifier };
use super::super::lexer::error::{ CompileError, ErrorType };

use super::ast_expr::*;
use super::ast::*;
use super::visitor::{ Transformable, Visitable, Transformer, Visitor };
use super::obj_type::*;

use std::any::Any;

pub type Program = Vec<FunctionNode>;

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

impl_transformable!(FunctionNode; vec params, result, implementation);
impl_visitable!(FunctionNode; vec params, result, implementation);
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

pub trait FunctionImplementationNode : Node
{
	fn dyn_clone(&self) -> Box<dyn FunctionImplementationNode>;
}

impl_partial_eq!(dyn FunctionImplementationNode);

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

impl_subnode!(FunctionImplementationNode for NativeFunctionNode);

impl_transformable!(NativeFunctionNode; );
impl_visitable!(NativeFunctionNode; );
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

impl_subnode!(FunctionImplementationNode for ImplementedFunctionNode);

impl_transformable!(ImplementedFunctionNode; body);
impl_visitable!(ImplementedFunctionNode; body);
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

impl_transformable!(ParameterNode; param_type);
impl_visitable!(ParameterNode; param_type);
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

impl Transformable for BlockNode
{
	fn transform(&mut self, transformer: &mut dyn Transformer)
	{
		for stmt in self.stmts.iter_mut() {
			take_mut::take(stmt, |node| transformer.transform_stmt(node));
		}
	}
}

impl_subnode!(StmtNode for BlockNode);

impl_visitable!(BlockNode; vec stmts);
impl_partial_eq!(BlockNode; stmts);

impl Clone for BlockNode {
	fn clone(&self) -> BlockNode {
		BlockNode {
			annotation: self.annotation.clone(),
			stmts: self.stmts.iter().map(|el| (**el).dyn_clone()).collect()
		}
	}
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

impl_subnode!(StmtNode for VariableDeclarationNode);

impl_transformable!(VariableDeclarationNode; variable_type, expr);
impl_visitable!(VariableDeclarationNode; variable_type, expr);
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

impl_subnode!(StmtNode for AssignmentNode);

impl_transformable!(AssignmentNode; assignee, expr);
impl_visitable!(AssignmentNode; assignee, expr);
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

impl_subnode!(StmtNode for ExprStmtNode);

impl_transformable!(ExprStmtNode; expr);
impl_visitable!(ExprStmtNode; expr);
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

impl_subnode!(StmtNode for IfNode);

impl_transformable!(IfNode; condition, block);
impl_visitable!(IfNode; condition, block);
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

impl_subnode!(StmtNode for WhileNode);

impl_transformable!(WhileNode; condition, block);
impl_visitable!(WhileNode; condition, block);
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

impl_subnode!(StmtNode for ReturnNode);

impl_transformable!(ReturnNode; expr);
impl_visitable!(ReturnNode; expr);
impl_partial_eq!(ReturnNode; expr);

pub trait TypeNode : Node + TypeDefinition {
	fn dyn_clone(&self) -> Box<dyn TypeNode>;
}

impl_partial_eq!(dyn TypeNode);

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

impl_transformable!(ArrTypeNode; base_type);
impl_visitable!(ArrTypeNode; base_type);
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

impl_subnode!(TypeNode for ArrTypeNode);

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

impl_subnode!(TypeNode for VoidTypeNode);

impl_transformable!(VoidTypeNode;);
impl_visitable!(VoidTypeNode;);
impl_partial_eq!(VoidTypeNode;);

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

impl TypeDefinition for ArrTypeNode
{
	fn calc_type(&self) -> Result<Option<Type>, CompileError>
	{
		if let Some(Type::Primitive(primitive_type)) = self.base_type.calc_type()? {
			return Ok(Some(Type::Array(primitive_type, self.dims as u32)));
		} else {
			return Err(CompileError::new(self.get_annotation().clone(), 
				"Arrays of void or non-primitive types are currently not supported".to_owned(), 
				ErrorType::IllegalArrayBaseType));
		}
	}
}

impl TypeDefinition for VoidTypeNode
{
	fn calc_type(&self) -> Result<Option<Type>, CompileError>
	{
		Ok(None)
	}
}