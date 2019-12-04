use super::super::lexer::tokens::{Identifier, Literal};
use super::super::lexer::position::TextPosition;
use super::super::lexer::error::CompileError;
use super::super::util::dyn_eq::DynEq;

use super::visitor::{ Visitable, Transformable, Visitor, Transformer };

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

pub fn cast<U: ?Sized + Node, T: Node>(node: Box<U>) -> Result<Box<T>, Box<U>> 
{
	if node.dynamic().is::<T>() {
		return Ok(node.dynamic_box().downcast().unwrap());
	} else {
		return Err(node);
	}
}

macro_rules! impl_transformable {
	($nodetype:ty; $($tail:tt)*) => {
		impl Transformable for $nodetype
		{
			fn transform(&mut self, f: &mut dyn Transformer)
			{
				take_mut::take(self, |node: $nodetype| *cast::<dyn Node, $nodetype>(f.transform(Box::new(node))).unwrap());
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