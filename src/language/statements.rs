use super::backend::{Backend, OutputError, Printable};
use super::identifier::Name;
use super::error::CompileError;
use super::position::TextPosition;
use super::AstNode;
use super::program::*;

use super::super::util::iterable::LifetimeIterable;

#[derive(Debug, Eq, Clone)]
pub struct If {
    pub pos: TextPosition,
    pub condition: Expression,
    pub body: Block,
}

#[derive(Debug, Eq, Clone)]
pub struct While {
    pub pos: TextPosition,
    pub condition: Expression,
    pub body: Block,
}

#[derive(Debug, Eq, Clone)]
pub struct Assignment {
    pub pos: TextPosition,
    pub assignee: Expression,
    pub value: Expression,
}

#[derive(Debug, Eq, Clone)]
pub struct LocalVariableDeclaration {
    pub declaration: Declaration,
    pub value: Option<Expression>,
}

#[derive(Debug, Eq, Clone)]
pub struct Return {
    pub pos: TextPosition,
    pub value: Option<Expression>,
}

#[derive(Debug, Eq, Clone)]
pub struct Label {
    pub pos: TextPosition,
    pub label: Name,
}

#[derive(Debug, Eq, Clone)]
pub struct Goto {
    pub pos: TextPosition,
    pub target: Name,
}


impl AstNode for If {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for If {
    fn eq(&self, rhs: &If) -> bool {
        self.condition == rhs.condition && self.body == rhs.body
    }
}

impl AstNode for While {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for While {
    fn eq(&self, rhs: &While) -> bool {
        self.condition == rhs.condition && self.body == rhs.body
    }
}

impl AstNode for Assignment {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for Assignment {
    fn eq(&self, rhs: &Assignment) -> bool {
        self.assignee == rhs.assignee && self.value == rhs.value
    }
}

impl AstNode for LocalVariableDeclaration {
    fn pos(&self) -> &TextPosition {
        self.declaration.pos()
    }
}

impl PartialEq for LocalVariableDeclaration {
    fn eq(&self, rhs: &LocalVariableDeclaration) -> bool {
        self.declaration == rhs.declaration && self.value == rhs.value
    }
}

impl AstNode for Return {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for Label {
    fn eq(&self, rhs: &Label) -> bool {
        self.label == rhs.label
    }
}

impl AstNode for Label {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for Return {
    fn eq(&self, rhs: &Return) -> bool {
        self.value == rhs.value
    }
}

impl PartialEq for Goto {
    fn eq(&self, rhs: &Goto) -> bool {
        self.target == rhs.target
    }
}

impl AstNode for Goto {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}


impl Statement for If {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn transform(self: Box<Self>, transformer: &mut dyn StatementTransformer) -> TransformerResult<dyn Statement> {
        transformer.transform_if(self)
    }

    fn transform_children(&mut self, transformer: &mut dyn StatementTransformer) -> Result<(), CompileError> {
        self.body.transform_children(transformer)
    }
}

impl Statement for While {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn transform(self: Box<Self>, transformer: &mut dyn StatementTransformer) -> TransformerResult<dyn Statement> {
        transformer.transform_while(self)
    }

    fn transform_children(&mut self, transformer: &mut dyn StatementTransformer) -> Result<(), CompileError> {
        self.body.transform_children(transformer)
    }
}

impl Statement for Return {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn transform(self: Box<Self>, transformer: &mut dyn StatementTransformer) -> TransformerResult<dyn Statement> {
        transformer.transform_return(self)
    }

    fn transform_children(&mut self, _transformer: &mut dyn StatementTransformer) -> Result<(), CompileError> {
        Ok(())
    }
}

impl Statement for Block {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn transform(self: Box<Self>, transformer: &mut dyn StatementTransformer) -> TransformerResult<dyn Statement> {
        transformer.transform_block(self)
    }

    fn transform_children(&mut self, transformer: &mut dyn StatementTransformer) -> Result<(), CompileError> {
        let mut error: Option<CompileError> = None;
        self.statements = self.statements.drain(..)
            .map(|s| s.transform(transformer))
            .map(|r| match r {
                Ok(s) => s,
                Err((s, e)) => {
                    if error.is_none() {
                        error = Some(e);
                    }
                    s
                }
            }).collect();
        return error.map(|e| Err(e)).unwrap_or(Ok(()));
    }
}

impl Statement for LocalVariableDeclaration {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn transform(self: Box<Self>, transformer: &mut dyn StatementTransformer) -> TransformerResult<dyn Statement> {
        transformer.transform_declaration(self)
    }

    fn transform_children(&mut self, _transformer: &mut dyn StatementTransformer) -> Result<(), CompileError> {
        Ok(())
    }
}

impl Statement for Assignment {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn transform(self: Box<Self>, transformer: &mut dyn StatementTransformer) -> TransformerResult<dyn Statement> {
        transformer.transform_assignment(self)
    }

    fn transform_children(&mut self, _transformer: &mut dyn StatementTransformer) -> Result<(), CompileError> {
        Ok(())
    }
}

impl Statement for Label {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn transform(self: Box<Self>, transformer: &mut dyn StatementTransformer) -> TransformerResult<dyn Statement> {
        transformer.transform_label(self)
    }

    fn transform_children(&mut self, _transformer: &mut dyn StatementTransformer) -> Result<(), CompileError> {
        Ok(())
    }
}

impl Statement for Goto {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }

    fn transform(self: Box<Self>, transformer: &mut dyn StatementTransformer) -> TransformerResult<dyn Statement> {
        transformer.transform_goto(self)
    }

    fn transform_children(&mut self, _transformer: &mut dyn StatementTransformer) -> Result<(), CompileError> {
        Ok(())
    }
}


impl Printable for If {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.print_if_header(self)?;
        self.body.print(printer)?;
        Ok(())
    }
}

impl Printable for While {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.print_while_header(self)?;
        self.body.print(printer)?;
        Ok(())
    }
}

impl Printable for LocalVariableDeclaration {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.print_declaration(self)
    }
}

impl Printable for Return {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.print_return(self)
    }
}

impl Printable for Assignment {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.print_assignment(self)
    }
}

impl Printable for Label {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.print_label(self)
    }
}

impl Printable for Goto {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.print_goto(self)
    }
}

impl<'a> LifetimeIterable<'a, Expression> for If {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(&self.condition))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.condition))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for While {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(&self.condition))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.condition))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Return {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        if let Some(ref val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        if let Some(ref mut val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }
}

impl<'a> LifetimeIterable<'a, Expression> for LocalVariableDeclaration {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        if let Some(ref val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        if let Some(ref mut val) = self.value {
            Box::new(std::iter::once(val))
        } else {
            Box::new(std::iter::empty())
        }
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Assignment {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(&self.assignee).chain(std::iter::once(&self.value)))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.assignee).chain(std::iter::once(&mut self.value)))
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Label {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Expression> for Goto {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for LocalVariableDeclaration {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for Assignment {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for Return {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for Goto {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for Label {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }
}

impl<'a> LifetimeIterable<'a, Block> for If {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(&self.body))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(&mut self.body))
    }
}

impl<'a> LifetimeIterable<'a, Block> for While {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(&self.body))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(&mut self.body))
    }
}
