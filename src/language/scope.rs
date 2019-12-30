use super::super::parser::prelude::*;
use super::obj_type::SymbolDefinition;

pub trait EnumerateDefinitions<'a> {
    type IntoIter: Iterator<Item = &'a dyn SymbolDefinition>;
    fn enumerate(self) -> Self::IntoIter;
}

pub trait Scope 
{
}

pub struct GlobalDefinitionsIter<'a> 
{
    iter: std::slice::Iter<'a, FunctionNode>
}

impl<'a> Iterator for GlobalDefinitionsIter<'a>
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|function| function as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a Program
{
    type IntoIter = GlobalDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        GlobalDefinitionsIter {
            iter: self.iter()
        }
    }
}

pub struct BlockNodeDefinitionsIter<'a>
{
    iter: std::slice::Iter<'a, Box<dyn StmtNode>>
}

impl<'a> Iterator for BlockNodeDefinitionsIter<'a>
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|stmt| stmt.dynamic().downcast_ref::<VariableDeclarationNode>()).map(|decl| decl as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a BlockNode 
{
    type IntoIter = BlockNodeDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        BlockNodeDefinitionsIter {
            iter: self.stmts.iter()
        }
    }
}

pub struct ParameterDefinitionsIter<'a>
{
    iter: std::slice::Iter<'a, Box<ParameterNode>>
}

impl<'a> Iterator for ParameterDefinitionsIter<'a>
{
    type Item = &'a dyn SymbolDefinition;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|param| &**param as &dyn SymbolDefinition)
    }
}

impl<'a> EnumerateDefinitions<'a> for &'a FunctionNode 
{
    type IntoIter = ParameterDefinitionsIter<'a>;

    fn enumerate(self) -> Self::IntoIter 
    {
        ParameterDefinitionsIter {
            iter: self.params.iter()
        }
    }
}

impl Scope for FunctionNode {}

impl Scope for Program {}

impl Scope for BlockNode {}

struct ScopeNode 
{
    definitions: Vec<Identifier>
}

fn test(d: &Vec<i8>) {
    d.into_iter();
}

impl ScopeNode 
{
    fn create<T>(scope: &T) -> ScopeNode 
        where for<'a> &'a T: EnumerateDefinitions<'a>
    {
        ScopeNode {
            definitions: scope.enumerate().map(|def| def.get_ident().clone()).collect()
        }
    }
}

pub struct ScopeStack 
{
    scope_stack: Option<Box<ScopeStack>>,
    scope_node: ScopeNode
}

impl ScopeStack {
    pub fn new(global: &Program) -> ScopeStack
    {
        ScopeStack {
            scope_stack: None,
            scope_node: ScopeNode::create(global)
        }
    }

    pub fn enter<T>(&mut self, scope: &T)
        where for<'a> &'a T: EnumerateDefinitions<'a>
    {
        let mut result: Box<ScopeStack> = Box::new(ScopeStack {
            scope_stack: std::mem::replace(&mut self.scope_stack, None),
            scope_node: ScopeNode::create(scope)
        });
        std::mem::swap(&mut self.scope_node, &mut result.scope_node);
        self.scope_stack = Some(result);
    }

    pub fn exit(&mut self) {
        let child = std::mem::replace(&mut self.scope_stack, None);
        *self = *child.expect("Cannot call exit() on empty scope stack");
    }
}

type TransformResultType = ();
type TransformFunctionType<T> = dyn FnMut(Box<T>) -> Box<T>;

pub trait Transformable<T: ?Sized>
{
	// If the closure does not return a object (i.e. if it panics), there is
	// an unrecoverable state, and the program will terminate
    fn transform(&mut self, f: &mut TransformFunctionType<T>) -> TransformResultType;
}

pub trait ScopedTransform 
{

}