use super::super::language::prelude::*;
use super::super::util::ref_eq::Ptr;

use std::hash::Hash;
use std::collections::{HashSet, HashMap};

///
/// Calculates a topological sorting of a directed, acyclic graph. The given iterator over 
/// nodes must yield at least a set of graph nodes from which every node is reachable, but
/// it is allowed to contain more graph nodes, possibly all nodes in the graph. The given edge
/// function should return all outgoing edge targets for a given node.
/// 
pub fn topological_sort<T: Sized, N, I, E>(nodes: N, mut edges: E) -> Result<Vec<T>, T>
    where N: Iterator<Item = T>, E: FnMut(T) -> I, I: Iterator<Item = T>, T: Copy + Hash + Eq
{
    fn insert_recursive<T: Sized, I, E>(node: T, result: &mut Vec<T>, path: &mut HashSet<T>, inserted_nodes: &mut HashSet<T>, edges: &mut E) -> Result<(), T>
        where E: FnMut(T) -> I, I: Iterator<Item = T>, T: Copy + Hash + Eq
    {
        if inserted_nodes.contains(&node) {
            return Ok(());
        }
        if path.contains(&node) {
            return Err(node);
        }
        path.insert(node);
        for child in edges(node) {
            insert_recursive(child, result, path, inserted_nodes, edges)?;
        }
        path.remove(&node);
        result.push(node);
        inserted_nodes.insert(node);
        return Ok(());
    }

    let mut result: Vec<T> = Vec::new();
    let mut inserted_nodes: HashSet<T> = HashSet::new();
    let mut path = HashSet::new();
    for node in nodes {
        if !inserted_nodes.contains(&node) {
            insert_recursive(node, &mut result, &mut path, &mut inserted_nodes, &mut edges)?;
            debug_assert!(path.len() == 0);
        }
    }
    result.reverse();
    return Ok(result);
}

struct CallData<'a> {
    called: HashSet<Ptr<'a, Function>>
}

fn collect_global_use_data_variable<'a>(
    ident: &'a Identifier, 
    parent_function: Ptr<'a, Function>, 
    global_scope: &DefinitionScopeStackConst<'_, 'a>, 
    result: &mut HashMap<Ptr<'a, Function>, CallData<'a>>
) {
    if ident.is_name() {
        if let Some(function) = global_scope.get(ident.unwrap_name()) {
            if let Some(use_data) = result.get_mut(&parent_function) {
                use_data.called.insert(Ptr::from(function.downcast::<Function>().unwrap()));
            } else {
                let mut use_data = CallData {
                    called: HashSet::new()
                };
                use_data.called.insert(Ptr::from(function.downcast::<Function>().unwrap()));
                result.insert(parent_function, use_data);
            }
        }
    }
}

impl CompileError {

    fn possible_recursion(function: &Function) -> CompileError {
        CompileError::new(
            function.pos(), 
            format!("Function {:?} can recurse", function.name), 
            ErrorType::Recursion)
    }
}

pub fn call_graph_topological_sort<'a>(program: &'a Program) -> Result<Vec<Ptr<'a, Function>>, CompileError> {
    let mut use_data: HashMap<Ptr<'a, Function>, CallData<'a>> = HashMap::new();
    let empty_call_data: CallData<'a> = CallData { called: HashSet::new() };
    program.for_functions(&mut |func: &'a Function, global_scopes| {
        func.traverse_preorder(global_scopes, &mut |block: &'a Block, _, _| {
            for statement in block.statements() {
                for expr in statement.expressions() {
                    for ident in expr.identifiers() {
                        collect_global_use_data_variable(ident, Ptr::from(func), global_scopes, &mut use_data);
                    }
                }
            }
            return RECURSE;
        })
    })?;
    return topological_sort(
        program.items().map(Ptr::from), 
        |fun| use_data.get(&fun).unwrap_or(&empty_call_data).called.iter().map(|target| *target)
    ).map_err(|f| CompileError::possible_recursion(&*f));
}

#[test]
fn test_topological_sort() {
    let ascending = topological_sort(vec![6, 3, 4, 1, 2, 9, 7, 0, 8].into_iter(), |x| x + 1..10);
    assert_eq!((0..10).collect::<Vec<_>>(), ascending.unwrap());
}

#[test]
fn test_topological_sort_cyclic() {
    let result = topological_sort(vec![0, 1, 2].into_iter(), |x| std::iter::once((x + 1) % 3));
    assert_eq!(Some(0), result.err());
}