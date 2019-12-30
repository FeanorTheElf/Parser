use super::super::lexer::position::TextPosition;
use super::super::util::dyn_eq::DynEq;

use super::visitor::{ Visitable, Transformable };

use std::fmt::Debug;
use std::any::Any;

pub type AstVec<T> = Vec<Box<T>>;
pub type Annotation = TextPosition;

pub trait Node : Debug + Any + DynEq + Visitable + Transformable
{
	fn get_annotation(&self) -> &Annotation;
	fn get_annotation_mut(&mut self) -> &mut Annotation;
	fn dyn_clone_node(&self) -> Box<dyn Node>;
	fn dynamic(&self) -> &dyn Any;
	fn dynamic_box(self: Box<Self>) -> Box<dyn Any>;
}

pub trait StmtNode : Node {
	fn dyn_clone(&self) -> Box<dyn StmtNode>;
}

impl_partial_eq!(dyn StmtNode);

pub trait UnaryExprNode : Node + Visitable 
{
	fn dyn_clone(&self) -> Box<dyn UnaryExprNode>;
}

impl_partial_eq!(dyn UnaryExprNode);

macro_rules! impl_transform {
	($self:ident; $transformer:ident; ) => {
		
	};
	($self:ident; $transformer:ident; ,) => {
		
	};
	($self:ident; $transformer:ident; vec $child:ident, $($tail:tt)*) => {
		{
			($self).$child.iter_mut().for_each(|node| {
				debug_assert_ne!(std::any::TypeId::of::<dyn StmtNode>(), node.dynamic().type_id());
				debug_assert_ne!(std::any::TypeId::of::<dyn UnaryExprNode>(), node.dynamic().type_id());
				node.transform($transformer)
			});
			impl_transform!($self; $transformer; $($tail)*);
		}
	};
	($self:ident; $transformer:ident; $child:ident, $($tail:tt)*) => {
		{
			debug_assert_ne!(std::any::TypeId::of::<dyn StmtNode>(), ($self).$child.dynamic().type_id());
			debug_assert_ne!(std::any::TypeId::of::<dyn UnaryExprNode>(), ($self).$child.dynamic().type_id());
			($self).$child.transform($transformer);
			impl_transform!($self; $transformer; $($tail)*);
		}
	};
}

macro_rules! impl_transformable {
	($nodetype:ty; $($tail:tt)*) => {
		impl Transformable for $nodetype 
		{
			#[allow(unused)]
			fn transform(&mut self, transformer: &mut dyn Transformer)
			{
				transformer.before(self);
				impl_transform!(self; transformer; $($tail)* ,);
				transformer.after(self);
			}
		}
	};
}

macro_rules! impl_visit {
	($self:ident; $visitor:ident; ) => {
		
	};
	($self:ident; $visitor:ident; ,) => {
		
	};
	($self:ident; $visitor:ident; vec $child:ident, $($tail:tt)*) => {
		{
			($self).$child.iter().try_for_each(|node| node.iterate($visitor))?;
			impl_visit!($self; $visitor; $($tail)*);
		}
	};
	($self:ident; $visitor:ident; $child:ident, $($tail:tt)*) => {
		{
			($self).$child.iterate($visitor)?;
			impl_visit!($self; $visitor; $($tail)*);
		}
	};
}

macro_rules! impl_visitable {
	($nodetype:ty; $($tail:tt)*) => {
		impl Visitable for $nodetype 
		{
			fn iterate<'a>(&'a self, visitor: &mut dyn Visitor<'a>) -> Result<(), CompileError>
			{
				visitor.enter(self)?;
				impl_visit!(self; visitor; $($tail)* ,);
				visitor.exit(self)?;
				return Ok(());
			}
		}
	};
}

macro_rules! impl_subnode {
	($supnode:ident for $subnode:ident) => {
				
		impl $supnode for $subnode 
		{
			fn dyn_clone(&self) -> Box<dyn $supnode> 
			{
				Box::new(self.clone())
			}
		}

	}
}

macro_rules! impl_node {
	($nodetype:ty) => {
		impl Node for $nodetype {
			fn get_annotation(&self) -> &Annotation
			{
				&self.annotation
			}

			fn get_annotation_mut(&mut self) -> &mut Annotation 
			{
				&mut self.annotation
			}

			fn dynamic(&self) -> &(dyn Any + 'static) 
			{
				self
			}

			fn dynamic_box(self: Box<Self>) -> Box<dyn Any + 'static>
			{
				self
			}

			fn dyn_clone_node(&self) -> Box<dyn Node> 
			{
				Box::new(self.clone())
			}
		}
	};
}