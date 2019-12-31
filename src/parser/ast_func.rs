use super::super::lexer::tokens::{ Identifier };
use super::super::lexer::error::{ CompileError, ErrorType };

use super::ast_expr::*;
use super::ast::*;
use super::print::Format;
use super::visitor::{ Transformable, Visitable, Transformer, Visitor };
use super::obj_type::*;

use std::any::Any;

use itertools::Itertools;

#[derive(Clone, Debug)]
pub struct Program
{
	annotation: Annotation,
	pub functions: Vec<FunctionNode>
}

impl Program 
{
	pub fn new(annotation: Annotation, mut data: AstVec<FunctionNode>) -> Self 
	{
		Program {
			annotation: annotation,
			functions: data.drain(..).map(|node| *node).collect()
		}
	}
}

impl Format for Program
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		for func in &self.functions {
			func.format(f, line_prefix)?;
			write!(f, "{}", line_prefix)?;
		}
		Ok(())
	}
}

impl_transformable!(Program; vec functions);
impl_visitable!(Program; vec functions);

impl PartialEq for Program
{
	fn eq(&self, rhs: &Program) -> bool
	{
		let mut rhs_groups: std::collections::HashMap<_, Vec<&FunctionNode>> = std::collections::HashMap::new();
		for (key, group) in &rhs.functions.iter().group_by(|f| &f.ident) {
			rhs_groups.insert(key, group.collect());
		}
		for (key, group) in &self.functions.iter().group_by(|f| &f.ident) {
			if let Some(rhs_group) = rhs_groups.get(key) {
				if !super::super::util::equal_ignore_order(&group.collect(), rhs_group) {
					return false;
				}
			} else {
				return false;
			}
		}
		return true;
	}
}

#[derive(Debug)]
pub struct FunctionNode 
{
	annotation: Annotation,
	pub ident: Identifier,
	pub params: AstVec<ParameterNode>,
	pub result: Box<dyn TypeNode>,
	pub implementation: Box<dyn FunctionImplementationNode>
}

impl FunctionNode 
{
	pub fn new(annotation: Annotation, ident: Identifier, params: AstVec<ParameterNode>, result: Box<dyn TypeNode>, implementation: Box<dyn FunctionImplementationNode>) -> Self {
		FunctionNode {
			annotation, ident, params, result, implementation
		}
	}
}

impl Format for FunctionNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		write!(f, "{}fn {}(", line_prefix, self.ident)?;
		for param in &self.params {
			param.format(f, line_prefix)?;
		}
		write!(f, "): ")?;
		self.result.format(f, line_prefix)?;
		write!(f, " ")?;
		self.implementation.format(f, line_prefix)
	}
}

impl_transformable!(FunctionNode; vec params, result, implementation);
impl_visitable!(FunctionNode; vec params, result, implementation);

impl PartialEq for FunctionNode
{
	fn eq(&self, rhs: &FunctionNode) -> bool
	{
		&self.ident == &rhs.ident && &self.params == &rhs.params && &self.result == &rhs.result && &self.implementation == &rhs.implementation
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

impl Format for NativeFunctionNode
{
	fn format(&self, f: &mut std::fmt::Formatter, _line_prefix: &str) -> std::fmt::Result
	{
		write!(f, "native")
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

impl Format for ImplementedFunctionNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		self.body.format(f, line_prefix)
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

impl Format for ParameterNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		write!(f, "{}: ", self.ident)?;
		self.param_type.format(f, line_prefix)?;
		write!(f, ", ")
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
		transformer.before(self);
		for stmt in self.stmts.iter_mut() {
			take_mut::take(stmt, |node| transformer.transform_stmt(node));
		}
		transformer.after(self);
	}
}

impl Format for BlockNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		write!(f, "{{")?;
		let new_line_prefix = line_prefix.to_owned() + "	";
		for stmt in &self.stmts {
			write!(f, "{}	", line_prefix)?;
			stmt.format(f, &new_line_prefix)?;
		}
		write!(f, "{}}}", line_prefix)
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
	pub expr: Option<Box<ExprNode>>
}

impl VariableDeclarationNode {
	pub fn new(annotation: Annotation, ident: Identifier, variable_type: Box<dyn TypeNode>, expr: Option<Box<ExprNode>>) -> Self {
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

impl Format for VariableDeclarationNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		write!(f, "let {}: ", self.ident)?;
		self.variable_type.format(f, line_prefix)?;
		if let Some(expr) = &self.expr {
			write!(f, " = ")?;
			expr.format(f, line_prefix)?;
		}
		write!(f, ";")
	}
}

impl_subnode!(StmtNode for VariableDeclarationNode);

impl_transformable!(VariableDeclarationNode; variable_type, opt expr);
impl_visitable!(VariableDeclarationNode; variable_type, opt expr);
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

impl Format for AssignmentNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		self.assignee.format(f, line_prefix)?;
		write!(f, " = ")?;
		self.expr.format(f, line_prefix)?;
		write!(f, ";")
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

impl Format for ExprStmtNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		self.expr.format(f, line_prefix)?;
		write!(f, ";")
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

impl Format for IfNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		write!(f, "if ")?;
		self.condition.format(f, line_prefix)?;
		write!(f, " ")?;
		self.block.format(f, line_prefix)
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

impl Format for WhileNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		write!(f, "while ")?;
		self.condition.format(f, line_prefix)?;
		write!(f, " ")?;
		self.block.format(f, line_prefix)
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

impl Format for ReturnNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		write!(f, "return ")?;
		self.expr.format(f, line_prefix)?;
		write!(f, ";")
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

impl Format for ArrTypeNode
{
	fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result
	{
		self.base_type.format(f, line_prefix)?;
		for _i in 0..self.dims {
			write!(f, "[]")?;
		}
		Ok(())
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
impl Format for VoidTypeNode
{
	fn format(&self, f: &mut std::fmt::Formatter, _line_prefix: &str) -> std::fmt::Result
	{
		write!(f, "void")
	}
}

impl_subnode!(TypeNode for VoidTypeNode);

impl_transformable!(VoidTypeNode;);
impl_visitable!(VoidTypeNode;);
impl_partial_eq!(VoidTypeNode;);

impl_display!(Program);
impl_display!(FunctionNode);
impl_display!(NativeFunctionNode);
impl_display!(ImplementedFunctionNode);
impl_display!(ParameterNode);
impl_display!(BlockNode);
impl_display!(VariableDeclarationNode);
impl_display!(AssignmentNode);
impl_display!(ExprStmtNode);
impl_display!(IfNode);
impl_display!(WhileNode);
impl_display!(ReturnNode);
impl_display!(ArrTypeNode);
impl_display!(VoidTypeNode);

impl_node!(Program);
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

#[cfg(test)]
use super::prelude::*;

#[cfg(test)]
impl ArrTypeNode
{
	pub fn test_val(dims: u8) -> Box<dyn TypeNode>
	{
		Box::new(ArrTypeNode::new(TextPosition::create(0, 0), Box::new(IntTypeNode::new(TextPosition::create(0, 0))), dims))
	}
}