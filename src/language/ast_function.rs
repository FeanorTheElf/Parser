use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::Name;
use super::ast::*;
use super::ast_statement::*;
use super::types::*;

#[derive(Debug)]
pub struct Function {
    pos: TextPosition,
    pub name: Name,
    pub parameters: Vec<Declaration>,
    pub return_type: Option<Type>,
    pub body: Option<Block>
}

impl PartialEq for Function {
    
    fn eq(&self, rhs: &Function) -> bool {
        self.name == rhs.name && self.parameters == rhs.parameters &&
        self.return_type == rhs.return_type && self.body == rhs.body
    }
}

impl AstNodeFuncs for Function {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for Function {}

impl Function {
    
    pub fn for_content<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStack, 
        f: &mut dyn FnMut(&'a Block, &DefinitionScopeStack) -> Result<(), CompileError>
    ) -> Result<(), CompileError> {
        let mut child_scope = parent_scopes.child_stack();
        for param in &self.parameters {
            child_scope.register(param.get_name().clone(), param);
        }
        if let Some(content) = &self.body {
            f(content, &child_scope)?;
        }
        return Ok(());
    }

    pub fn for_content_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut, 
        f: &mut dyn FnMut(&mut Block, &DefinitionScopeStackMut) -> Result<(), CompileError>
    ) -> Result<(), CompileError> {
        let mut child_scope = parent_scopes.child_stack();
        for param in &mut self.parameters {
            child_scope.register(param.get_name().clone(), param);
        }
        if let Some(content) = &mut self.body {
            f(content, &child_scope)?;
        }
        return Ok(());
    }
    
    pub fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStack, 
        f: &mut dyn FnMut(&'a Block, &DefinitionScopeStack) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.for_content(parent_scopes, &mut |content: &'a Block, scopes| {
            content.traverse_preorder(scopes, f)
        })
    }

    pub fn traverse_preorder_mut(
        &mut self, 
        parent_scopes: &DefinitionScopeStackMut, 
        f: &mut dyn FnMut(&mut Block, &DefinitionScopeStackMut) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.for_content_mut(parent_scopes, &mut |content: &mut Block, scopes| {
            content.traverse_preorder_mut(scopes, f)
        })
    }
}

impl SymbolDefinitionFuncs for Function {

    fn get_name(&self) -> &Name {
        &self.name
    }

    fn cast_statement_mut(&mut self) -> Option<&mut dyn Statement> {
        None
    }
}

impl SymbolDefinition for Function {}

impl SiblingSymbolDefinitionFuncs for Function {

    fn is_backward_visible(&self) -> bool {
        true
    }
}

impl SiblingSymbolDefinition for Function {}

impl Function {

    #[cfg(test)]
    pub fn new<const N: usize>(name: &'static str, params: [(&'static str, Type); N], return_type: Option<Type>, body: Block) -> Function {
        Function {
            pos: TextPosition::NONEXISTING,
            name: Name::l(name),
            parameters: params.iter().map(|(name, ty)| Declaration {
                name: Name::l(name),
                var_type: ty.clone(),
                pos: TextPosition::NONEXISTING
            }).collect(),
            return_type: return_type,
            body: Some(body)
        }
    }
}

#[derive(Debug)]
pub struct Program {
    pub items: Vec<Function>
}

#[derive(Debug)]
enum FunctionMutOrPlaceholder<'a> {
    Function(&'a mut Function),
    PlaceholderFunctionName(Name)
}

impl Program {
    
    pub fn for_functions<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStack, 
        f: &mut dyn FnMut(&'a Function, &DefinitionScopeStack) -> Result<(), CompileError>
    ) -> Result<(), CompileError> {
        let mut child_scope = parent_scopes.child_stack();
        for item in &self.items {
            if item.is_backward_visible() {
                child_scope.register(item.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic(item));
            }
        }
        for item in &self.items {
            f(item, &child_scope)?;
            if !item.is_backward_visible() {
                child_scope.register(item.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic(item));
            }
        }
        return Ok(());
    }

    pub fn for_functions_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut, 
        f: &mut dyn FnMut(&mut Function, &DefinitionScopeStackMut) -> Result<(), CompileError>
    ) -> Result<(), CompileError> {
        // the idea is the same as in `traverse_preorder_mut()`, it is just a bit
        // simpler because there is no polymorphism in the items
        let mut child_scope = parent_scopes.child_stack();
        let mut data = Vec::new();
        for item in &mut self.items {
            if item.is_backward_visible() {
                data.push(FunctionMutOrPlaceholder::PlaceholderFunctionName(item.get_name().clone()));
                child_scope.register(item.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic_mut(item));
            } else {
                data.push(FunctionMutOrPlaceholder::Function(item));
            }
        }
        for item in data.into_iter() {
            match item {
                FunctionMutOrPlaceholder::Function(item) => {
                    f(item, &child_scope)?;
                    child_scope.register(item.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic_mut(item));
                },
                FunctionMutOrPlaceholder::PlaceholderFunctionName(name) => {
                    let item = child_scope.unregister(&name).downcast_mut::<Function>().unwrap();
                    f(item, &child_scope)?;
                    child_scope.register(name, <_ as SymbolDefinitionDynCastable>::dynamic_mut(item));
                }
            }
        }
        return Ok(());
    }
}