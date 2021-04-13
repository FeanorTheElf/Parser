use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::Name;
use super::ast::*;
use super::ast_expr::*;
use super::types::*;
use super::scopes::*;
use super::symbol::*;

pub trait StatementFuncs: AstNode {
    ///
    /// Iterates over all blocks contained directly in this statement.
    /// 
    /// # Details
    /// This will not yield nested blocks. If this statement is a block itself, it will only return itself.
    /// The order in which the blocks are returned is unspecified.
    /// 
    /// # Example
    /// ```
    /// let a = Block::parse(&mut fragment_lex("{{{}}{}}"), &mut TypeVec::new()).unwrap();
    /// assert_eq!(a.subblocks().count(), 1); // returns only the top level block
    /// assert_eq!(a.subblocks().flat_map(|b| b.statements.iter()).flat_map(|s| s.subblocks()).count(), 2); // returns the proper subblocks, also non-nested
    /// ```
    /// 
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)>;
    ///
    /// See `subblocks()`
    /// 
    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)>;

    ///
    /// Iterates over all expressions contained directly in this statement.
    /// 
    /// # Details
    /// This will not yield nested expressions. If this statement is only one expression, only the statement itself will be returned.
    /// The expressions will be returned in the order of execution. If this statement contains other statements that in turn contain
    /// expressions, these expressions will not be returned.
    /// 
    /// # Example
    /// ```
    /// let a = Statement::parse(&mut fragment_lex("a = b(c, 0,);"), &mut TypeVec::new()).unwrap();
    /// assert_eq!(a.expressions().count(), 2); // returns only the top level expressions 'a' and 'b(c, 0,)', in this order
    /// ```
    /// 
    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)>;
    ///
    /// See `expressions()`
    /// 
    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)>;

    ///
    /// Iterates over all names contained in this statement.
    /// 
    /// # Details
    /// This will return all names (i.e. user-defined identifiers) in this statement, which includes uses of variables/functions/... and
    /// also declarations. Since names cannot be nested, all names are returned (as opposed to subblocks() and expressions()). The order
    /// in which the names are returned is unspecified.
    ///
    /// # Example
    /// ```
    /// let a = Statement::parse(&mut fragment_lex("let a: int = b(c, 0,);"), &mut TypeVec::new()).unwrap();
    /// assert_eq!(a.names().map(|n| n.name.as_str()).collect::<Vec<_>>(), vec!["a", "c", "b"]);
    /// ```
    /// 
    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(self.expressions().flat_map(|e| e.names()))
    }

    ///
    /// See `names()`
    /// 
    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(self.expressions_mut().flat_map(|e| e.names_mut()))
    }

    ///
    /// Calls the given function on all statements contained in this statement. These statements form
    /// a tree, which is traversed in preorder. Additionally, the callback function is given
    /// scope stack with all outer definitions that are visible from the current statement.
    /// 
    /// # Details
    /// 
    /// When called on a statement, the statement itself is the first element the callback is called with.
    /// After that, the search will then continue with the statements in the block
    /// 
    /// When the callback is called on a statement, the scope stack should therefore contain all symbols
    /// that are defined outside the statement and are either visible before declaration, or declared
    /// before the statement in the parent block.
    /// 
    /// By returning `RECURSE` resp. `DONT_RECURSE`, one can control whether the search continues
    /// in potential subblocks of the current statement. If `DONT_RECURSE` is returned, the whole
    /// subtree whose root is the current statement is skipped.
    /// 
    /// # Example
    /// ```
    /// let a = Block::parse(&mut fragment_lex("{
    ///     let a: int = 1;
    ///     if (a == 1) {
    ///         let b: int = 2;
    ///     }
    /// }"), &mut ParserContext::new()).unwrap();
    /// let mut counter = 0;
    /// let scopes = ScopeStack::new();
    /// a.traverse_preorder(&scopes, &mut |b, s| {
    ///     counter += 1;
    ///     if counter > 1 { // the if
    ///         assert!(s.get(&Name::l("a")).is_some());
    ///         assert!(s.get(&Name::l("b")).is_none());
    ///     } else {
    ///         assert!(s.get(&Name::l("a")).is_none());
    ///     }
    /// });
    /// assert_eq!(3, counter);
    /// ```
    /// 
    fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError>;

    ///
    /// See `traverse_preorder()`
    /// 
    fn traverse_preorder_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError>;

    ///
    /// Similar to `traverse_preorder()`, however instead of a statement, the closure is called
    /// for each block contained in the statement. In particular, for many statements this will 
    /// not call the callback at all. For blocks, the callback will be called once for the whole block,
    /// and then for each subblock in the scope tree.
    /// 
    /// For details, see `traverse_preorder()`.
    /// 
    fn traverse_preorder_block<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a Block, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.traverse_preorder(parent_scopes, &mut |statement, scopes| {
            let mut subblocks = statement.subblocks();
            if let Some(block) = subblocks.next() {
                assert!(subblocks.next().is_none());
                return f(block, scopes);
            } else {
                return DONT_RECURSE;
            }
        })
    }

    ///
    /// See `traverse_preorder_block()`.
    /// 
    fn traverse_preorder_block_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut Block, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        self.traverse_preorder_mut(parent_scopes, &mut |statement, scopes| {
            let mut subblocks = statement.subblocks_mut();
            if let Some(block) = subblocks.next() {
                assert!(subblocks.next().is_none());
                return f(block, scopes);
            } else {
                return DONT_RECURSE;
            }
        })
    }

    ///
    /// If this statement defines a symbol that is visible in the whole parent scope
    /// (i.e. also by sibling statements), this returns the corresponding symbol
    /// definition object. If no such symbol is defined by the current statement, this
    /// returns None.
    /// For mutable access to that object, see `as_sibling_symbol_definition_mut()`.
    /// 
    fn as_sibling_symbol_definition(&self) -> Option<&dyn SiblingSymbolDefinition> {
        None
    }

    ///
    /// See `as_sibling_symbol_definition`.
    /// 
    fn as_sibling_symbol_definition_mut(&mut self) -> Option<&mut dyn SiblingSymbolDefinition> {
        None
    }
}

dynamic_subtrait!{ Statement: StatementFuncs; StatementDynCastable}

///
/// Describes symbol definitions which are visible in a scope to which they
/// belong themselves, i.e. are visible from "sibling" items.
/// 
pub trait SiblingSymbolDefinitionFuncs: SymbolDefinition {
    ///
    /// Returns true if the current symbol definition is visible from items
    /// that come before it in source code.
    /// 
    fn is_backward_visible(&self) -> bool;
}

dynamic_subtrait!{ SiblingSymbolDefinition: SiblingSymbolDefinitionFuncs; SiblingSymbolDefinitionDynCastable }

#[derive(Debug)]
pub struct Block {
    pos: TextPosition,
    statements: Vec<Box<dyn Statement>>
}

impl PartialEq for Block {
    fn eq(&self, rhs: &Block) -> bool {
        if self.statements.len() != rhs.statements.len() {
            return false;
        }
        self.statements.iter().zip(rhs.statements.iter()).all(|(a, b)| a.dyn_eq(b))
    }
}

impl AstNodeFuncs for Block {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for Block {}

#[derive(Debug)]
enum StatementMutOrPlaceholder<'a> {
    Statement(&'a mut dyn Statement),
    PlaceholderDefOfName(Name)
}

impl StatementFuncs for Block {

    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(self))
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(self))
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(self.statements.iter().flat_map(|s| s.names()))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(self.statements.iter_mut().flat_map(|s| s.names_mut()))
    }

    fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        let recurse = f(self, parent_scopes);
        self.traverse_preorder_base(parent_scopes, f, recurse)
    }

    fn traverse_preorder_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        let recurse = f(self, parent_scopes);
        self.traverse_preorder_mut_base(parent_scopes, f, recurse)
    }
}

impl Block {
    
    pub fn traverse_preorder_base<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult,
        recurse: TraversePreorderResult
    ) -> Result<(), CompileError> {
        match recurse {
            Err(TraversePreorderCancel::RealError(e)) => return Err(e),
            Err(TraversePreorderCancel::DoNotRecurse) => {},
            Ok(_) => {
                let mut child_scope = parent_scopes.child_stack();
                for statement in &self.statements {
                    if statement.as_sibling_symbol_definition().is_some() {
                        let def = statement.as_sibling_symbol_definition().unwrap();
                        if def.is_backward_visible() {
                            child_scope.register(def.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic(def));
                        }
                    }
                }
                for statement in &self.statements {
                    statement.traverse_preorder(&child_scope, f)?;
                    if statement.as_sibling_symbol_definition().is_some() {
                        let def = statement.as_sibling_symbol_definition().unwrap();
                        if !def.is_backward_visible() {
                            child_scope.register(def.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic(def));
                        }
                    }
                }
            }
        };
        return Ok(());
    }

    pub fn traverse_preorder_mut_base<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult,
        recurse: TraversePreorderResult
    ) -> Result<(), CompileError> {
        match recurse {
            Err(TraversePreorderCancel::RealError(e)) => return Err(e),
            Err(TraversePreorderCancel::DoNotRecurse) => {},
            Ok(_) => {
                // this is a little bit more complicated due to single mutable reference
                // constraints: Add all symbol definitions to the scope stack, and keep
                // the other statement refs in a temporary vector.
                // Then process the statements from the vector, and when we encounter a 
                // symbol definition, temporarily take the reference from the scope stack,
                // use it and then put it back 
                let mut child_scope = parent_scopes.child_stack();
                let mut data = Vec::new();
                for statement in &mut self.statements {
                    if statement.as_sibling_symbol_definition_mut().is_some() &&
                        statement.as_sibling_symbol_definition_mut().unwrap().is_backward_visible() 
                    {
                        let def = statement.as_sibling_symbol_definition_mut().unwrap();
                        data.push(StatementMutOrPlaceholder::PlaceholderDefOfName(def.get_name().clone()));
                        child_scope.register(def.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic_mut(def));
                    } else {
                        data.push(StatementMutOrPlaceholder::Statement(&mut **statement));
                    }
                }
                for statement in data.into_iter() {
                    match statement {
                        StatementMutOrPlaceholder::Statement(statement) => {
                            statement.traverse_preorder_mut(&child_scope, &mut *f)?;
                            // in case this is a sibling symbol definition and found here, it must be 
                            // forward-only visible, so add it now to the scope stack to be visible in
                            // subsequent traversals
                            if statement.as_sibling_symbol_definition_mut().is_some() {
                                let def = statement.as_sibling_symbol_definition_mut().unwrap();
                                child_scope.register(def.get_name().clone(), <_ as SymbolDefinitionDynCastable>::dynamic_mut(def));
                            }
                        },
                        StatementMutOrPlaceholder::PlaceholderDefOfName(name) => {
                            let statement = child_scope.unregister(&name).cast_statement_mut().unwrap();
                            statement.traverse_preorder_mut(&child_scope, &mut *f)?;
                            let def = statement.as_sibling_symbol_definition_mut().unwrap();
                            child_scope.register(name, <_ as SymbolDefinitionDynCastable>::dynamic_mut(def));
                        }
                    }
                }
            }
        };
        return Ok(());
    }
}

impl Statement for Block {}

impl Block {

    pub fn statements(&self) -> impl Iterator<Item = &dyn Statement> {
        self.statements.iter().map(|x| &**x)
    }

    pub fn statements_mut(&mut self) -> impl Iterator<Item = &mut dyn Statement> {
        self.statements.iter_mut().map(|x| &mut **x)
    }

    #[cfg(test)]
    pub fn test<const N: usize>(statements: [Box<dyn Statement>; N]) -> Block {
        Block {
            pos: TextPosition::NONEXISTING,
            statements: std::array::IntoIter::new(statements).collect()
        }
    }

    pub fn new(pos: TextPosition, statements: Vec<Box<dyn Statement>>) -> Self {
        Block {
            pos, statements
        }
    }
}

#[derive(Debug)]
pub struct Declaration {
    pub pos: TextPosition,
    pub name: Name,
    pub var_type: Type
}

#[derive(Debug, PartialEq)]
pub struct LocalVariableDeclaration {
    pub declaration: Declaration,
    pub value: Expression
}

impl PartialEq for Declaration {
    fn eq(&self, rhs: &Declaration) -> bool {
        self.name == rhs.name && self.var_type == rhs.var_type
    }
}

impl AstNodeFuncs for Declaration {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for Declaration {}

impl SymbolDefinitionFuncs for Declaration {

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
        &self.var_type
    }
}

impl SymbolDefinition for Declaration {}

impl AstNodeFuncs for LocalVariableDeclaration {

    fn pos(&self) -> &TextPosition {
        self.declaration.pos()
    }
}

impl AstNode for LocalVariableDeclaration {}

impl StatementFuncs for LocalVariableDeclaration {
    
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(&self.value))
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(&mut self.value))
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(
            std::iter::once(&self.declaration.name).chain(
            self.value.names()
        ))
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(
            std::iter::once(&mut self.declaration.name).chain(
            self.value.names_mut()
        ))
    }

    fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        f(self, parent_scopes).ignore_cancel()
    }

    fn traverse_preorder_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        f(self, parent_scopes).ignore_cancel()
    }

    fn as_sibling_symbol_definition(&self) -> Option<&dyn SiblingSymbolDefinition> {
        Some(self)
    }

    fn as_sibling_symbol_definition_mut(&mut self) -> Option<&mut dyn SiblingSymbolDefinition> {
        Some(self)
    }
}

impl Statement for LocalVariableDeclaration {}

impl SymbolDefinitionFuncs for LocalVariableDeclaration {

    fn get_name(&self) -> &Name {
        self.declaration.get_name()
    }

    fn get_name_mut(&mut self) -> &mut Name {
        self.declaration.get_name_mut()
    }

    fn cast_statement_mut(&mut self) -> Option<&mut dyn Statement> {
        Some(self)
    }

    fn get_type(&self) -> &Type {
        self.declaration.get_type()
    }
}

impl SymbolDefinition for LocalVariableDeclaration {}

impl SiblingSymbolDefinitionFuncs for LocalVariableDeclaration {

    fn is_backward_visible(&self) -> bool {
        false
    }
}

impl SiblingSymbolDefinition for LocalVariableDeclaration {}

impl LocalVariableDeclaration {

    #[cfg(test)]
    pub fn new(name: &'static str, var_type: Type, value: Expression) -> LocalVariableDeclaration {
        LocalVariableDeclaration {
            declaration: Declaration {
                pos: TextPosition::NONEXISTING,
                name: Name::l(name),
                var_type: var_type
            },
            value: value
        }
    }
}

impl StatementFuncs for Expression {
    
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::empty())
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(std::iter::once(self))
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(std::iter::once(self))
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(self.names())
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(self.names_mut())
    }

    fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        f(self, parent_scopes).ignore_cancel()
    }

    fn traverse_preorder_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        f(self, parent_scopes).ignore_cancel()
    }
}

impl Statement for Expression {}

#[test]
fn test_block_preorder_traversal_mut() {
    let mut block = Block::test([
        Box::new(LocalVariableDeclaration::new("a", SCALAR_INT, Expression::var("x"))), 
        Box::new(Block::test([])),
        Box::new(LocalVariableDeclaration::new("b", SCALAR_INT, Expression::var("x"))),
        Box::new(Block::test([]))
    ]);
    let mut counter = 0;
    let mut callback = |statement: &mut dyn Statement, scopes: &DefinitionScopeStackMut| {
        if statement.downcast::<Block>().is_none() {
            return RECURSE;
        }
        if counter % 3 == 0 {
            assert!(scopes.get(&Name::l("a")).is_none());
        } else if counter % 3 == 1 {
            assert!(scopes.get(&Name::l("a")).is_some());
            assert!(scopes.get(&Name::l("b")).is_none());
        } else {
            assert!(scopes.get(&Name::l("b")).is_some());
        }
        counter += 1;
        return RECURSE;
    };

    block.traverse_preorder_mut(&DefinitionScopeStackMut::new(), &mut callback).unwrap();

    assert_eq!(3, counter);
}