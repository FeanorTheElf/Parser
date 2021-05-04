use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::Name;
use super::ast::*;
use super::ast_statement::*;
use super::types::*;
use super::scopes::*;
use super::symbol::*;

use super::super::util::ref_eq::Ptr;

#[derive(Debug)]
pub struct Function {
    pos: TextPosition,
    pub name: Name,
    pub parameters: Vec<Declaration>,
    pub body: Option<Block>,
    function_type: Type
}

impl Function {

    pub fn new(pos: TextPosition, name: Name, parameters: Vec<Declaration>, return_type: Option<Type>, body: Option<Block>) -> Function {
        Function {
            pos: pos,
            name: name,
            function_type: Type::function_type(parameters.iter().map(|d| d.get_type().clone()), return_type),
            parameters: parameters,
            body: body
        }
    }

    pub fn function_type(&self) -> &FunctionType {
        self.function_type.as_function().unwrap()
    }

    pub fn function_type_mut(&mut self) -> &mut FunctionType {
        self.function_type.as_function_mut().unwrap()
    }

    pub fn return_type(&self) -> Option<&Type> {
        self.function_type().return_type()
    }
}

impl PartialEq for Function {
    
    fn eq(&self, rhs: &Function) -> bool {
        self.name == rhs.name && self.parameters == rhs.parameters &&
        self.function_type == rhs.function_type && self.body == rhs.body
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
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a Block, &DefinitionScopeStackConst<'_, 'a>, &FunctionType) -> Result<(), CompileError>
    ) -> Result<(), CompileError> {
        let mut child_scope = parent_scopes.child_stack();
        for param in &self.parameters {
            child_scope.register(param.get_name().clone(), param);
        }
        if let Some(content) = &self.body {
            f(content, &child_scope, self.function_type())?;
        }
        return Ok(());
    }

    pub fn for_content_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, 'a>, 
        f: &mut dyn FnMut(&'a mut Block, &DefinitionScopeStackMut<'_, 'a>, &mut FunctionType) -> Result<(), CompileError>
    ) -> Result<(), CompileError> {
        let mut child_scope = parent_scopes.child_stack();
        for param in &mut self.parameters {
            child_scope.register(param.get_name().clone(), param);
        }
        if let Some(content) = &mut self.body {
            f(content, &child_scope, self.function_type.as_function_mut().unwrap())?;
        }
        return Ok(());
    }
    
    ///
    /// See the corresponding method on Statement
    /// 
    pub fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>, &FunctionType) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.for_content(parent_scopes, &mut |content: &'a Block, scopes, ty| {
            content.traverse_preorder(scopes, &mut |c, s| f(c, s, ty))
        })
    }

    ///
    /// See the corresponding method on Statement
    /// 
    pub fn traverse_preorder_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>, &mut FunctionType) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.for_content_mut(parent_scopes, &mut |content: &mut Block, scopes, ty| {
            content.traverse_preorder_mut(scopes, &mut |c, s| f(c, s, ty))
        })
    }

    ///
    /// See the corresponding method on Statement
    /// 
    pub fn traverse_preorder_block<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a Block, &DefinitionScopeStackConst<'_, 'a>, &FunctionType) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.for_content(parent_scopes, &mut |content: &'a Block, scopes, ty| {
            content.traverse_preorder_block(scopes, &mut |c, s| f(c, s, ty))
        })
    }

    ///
    /// See the corresponding method on Statement
    /// 
    pub fn traverse_preorder_block_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut Block, &DefinitionScopeStackMut<'_, '_>, &mut FunctionType) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.for_content_mut(parent_scopes, &mut |content: &mut Block, scopes, ty| {
            content.traverse_preorder_block_mut(scopes, &mut |c, s| f(c, s, ty))
        })
    }
}

impl SymbolDefinitionFuncs for Function {

    fn get_name(&self) -> &Name {
        &self.name
    }

    fn get_name_mut(&mut self) -> &mut Name {
        &mut self.name
    }

    fn cast_statement_mut(&mut self) -> Option<&mut dyn Statement> {
        None
    }

    fn get_type(&self) -> &Type {
        &self.function_type
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
    pub fn test<const N: usize>(name: &'static str, params: [(&'static str, Type); N], return_type: Option<Type>, body: Block) -> Function {
        Function::new(
            TextPosition::NONEXISTING,
            Name::l(name),
            params.iter().map(|(name, ty)| Declaration {
                name: Name::l(name),
                var_type: ty.clone(),
                pos: TextPosition::NONEXISTING
            }).collect(),
            return_type,
            Some(body)
        )
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
        f: &mut dyn FnMut(&'a Function, &DefinitionScopeStackConst<'_, 'a>) -> Result<(), CompileError>
    ) -> Result<(), CompileError> {
        self.for_functions_stored_order(self.items(), f)
    }

    pub fn for_functions_mut<'a>(
        &'a mut self, 
        f: &mut dyn FnMut(&mut Function, &DefinitionScopeStackMut<'_, '_>) -> Result<(), CompileError>
    ) -> Result<(), CompileError> {
        self.for_functions_stored_order_mut(0..self.items.len(), f)
    }

    pub fn for_functions_ordered<'a, F>(
        &'a self,
        order: F,
        f: &mut dyn FnMut(&'a Function, &DefinitionScopeStackConst<'_, 'a>) -> Result<(), CompileError>
    ) -> Result<(), CompileError> 
        where F: FnOnce(&Program) -> Result<Vec<Ptr<Function>>, CompileError>
    {
        self.for_functions_stored_order(order(self)?.into_iter().map(Ptr::get), f)
    }

    pub fn for_functions_ordered_mut<'a, F>(
        &'a mut self,
        order: F,
        f: &mut dyn FnMut(&mut Function, &DefinitionScopeStackMut<'_, '_>) -> Result<(), CompileError>
    ) -> Result<(), CompileError> 
        where F: FnOnce(&Program) -> Result<Vec<Ptr<Function>>, CompileError>
    {
        let order_indices = self.get_function_order_indices(order(self)?).collect::<Vec<_>>();
        self.for_functions_stored_order_mut(order_indices.into_iter(), f)
    }

    fn get_function_order_indices<'a>(
        &'a self,
        order: Vec<Ptr<'a, Function>>
    ) -> impl 'a + Iterator<Item = usize> 
    {
        self.items.iter().map(move |f: &Function| order.iter().enumerate().find(|(_, g)| g.get_name() == f.get_name()).unwrap().0)
    }

    fn for_functions_stored_order<'a, I>(
        &'a self,
        order: I,
        f: &mut dyn FnMut(&'a Function, &DefinitionScopeStackConst<'_, 'a>) -> Result<(), CompileError>
    ) -> Result<(), CompileError> 
        where I: Iterator<Item = &'a Function>
    {
        let mut child_scope = DefinitionScopeStackConst::new();
        for item in &self.items {
            // other things should not exist and are not implemented
            if !item.is_backward_visible() {
                unimplemented!();
            }
            child_scope.register(item.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic(item));
        }
        for item in order {
            f(item, &child_scope)?;
        }
        return Ok(());
    }

    ///
    /// Iterates over all items in the order specified by the given iterator. The iterator
    /// should have exactly the same length as there are items, and assigns each item a
    /// unique number according to which the items are sorted prior to iteration.
    /// 
    /// For the non-order-related contract, see `for_functions_mut()`.
    /// 
    fn for_functions_stored_order_mut<'a, I>(
        &'a mut self,
        mut order: I,
        f: &mut dyn FnMut(&mut Function, &DefinitionScopeStackMut<'_, '_>) -> Result<(), CompileError>
    ) -> Result<(), CompileError> 
        where I: Iterator<Item = usize>
    {
        // the idea is the same as in `traverse_preorder_mut()`, it is just a bit
        // simpler because there is no polymorphism in the items
        let mut child_scope = DefinitionScopeStackMut::new();
        let mut data = Vec::new();
        let mut item_iter = self.items.iter_mut();
        for (item, order_index) in (&mut item_iter).zip(&mut order) {
            // other things currently not exist and are not implemented
            if !item.is_backward_visible() {
                unimplemented!();
            }
            data.push((order_index, item.get_name().clone()));
            child_scope.register(item.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic_mut(item));
        }
        assert!(item_iter.next().is_none());
        assert!(order.next().is_none());

        data.sort_unstable_by_key(|(order_index, _)| *order_index);

        for (_order_index, name) in data.into_iter() {
            let item = child_scope.unregister(&name).downcast_mut::<Function>().unwrap();
            f(item, &child_scope)?;
            child_scope.register(name, <_ as SymbolDefinitionDynCastable>::dynamic_mut(item));
        }
        return Ok(());
    }

    ///
    /// See the corresponding method on Statement
    /// 
    pub fn traverse_preorder<'a>(
        &'a self, 
        f: &'a mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.for_functions(&mut |function, scopes| function.for_content(scopes, &mut |body, scopes, _| {
            body.traverse_preorder(scopes, f)
        }))
    }

    ///
    /// See the corresponding method on Statement
    /// 
    pub fn traverse_preorder_mut<'a>(
        &'a mut self, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.for_functions_mut(&mut |function, scopes| function.for_content_mut(scopes, &mut |body, scopes, _| {
            body.traverse_preorder_mut(scopes, f)
        }))
    }

    ///
    /// See the corresponding method on Statement
    /// 
    pub fn traverse_preorder_block<'a>(
        &'a self, 
        f: &mut dyn FnMut(&'a Block, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.for_functions(&mut |function, scopes| function.for_content(scopes, &mut |body, scopes, _| {
            body.traverse_preorder_block(scopes, f)
        }))
    }

    ///
    /// See the corresponding method on Statement
    /// 
    pub fn traverse_preorder_block_mut<'a>(
        &'a mut self, 
        f: &mut dyn FnMut(&mut Block, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.for_functions_mut(&mut |function, scopes| function.for_content_mut(scopes, &mut |body, scopes, _| {
            body.traverse_preorder_block_mut(scopes, f)
        }))
    }

    pub fn items(&self) -> impl Iterator<Item = &Function> {
        self.items.iter()
    }

    pub fn items_mut(&mut self) -> impl Iterator<Item = &mut Function> {
        self.items.iter_mut()
    }
}

#[cfg(test)]
use super::super::parser::TopLevelParser;
#[cfg(test)]
use super::super::lexer::lexer::lex_str;

#[test]
fn test_for_functions_ordered() {
    let mut program = Program::parse(&mut lex_str("
    
        fn foo() {}
        fn bar() {}
        fn foobar() {}
        fn baz() {}
    
    ")).unwrap();

    let mut names = Vec::new();
    program.for_functions(&mut |f: &Function, _| {
        names.push(f.get_name().clone());
        return Ok(());
    }).unwrap();
    assert_eq!(names, vec!["foo", "bar", "foobar", "baz"]);

    names.clear();
    program.for_functions_mut(&mut |f, _| {
        names.push(f.get_name().clone());
        return Ok(());
    }).unwrap();
    assert_eq!(names, vec!["foo", "bar", "foobar", "baz"]);

    names.clear();
    program.for_functions_ordered(
        |prog| Ok([&prog.items[0], &prog.items[2], &prog.items[1], &prog.items[3]].iter().map(|x| Ptr::from(*x)).collect()),
        &mut |f, _| {
            names.push(f.get_name().clone());
            return Ok(());
        }
    ).unwrap();
    assert_eq!(names, vec!["foo", "foobar", "bar", "baz"]);

    names.clear();
    program.for_functions_ordered_mut(
        |prog| Ok([&prog.items[0], &prog.items[2], &prog.items[1], &prog.items[3]].iter().map(|x| Ptr::from(*x)).collect()),
        &mut |f, _| {
            names.push(f.get_name().clone());
            return Ok(());
        }
    ).unwrap();
    assert_eq!(names, vec!["foo", "foobar", "bar", "baz"]);
}