use super::super::language::prelude::*;
use super::super::analysis::scope::NameScopeStack;
use std::collections::{HashMap, HashSet};

struct NameMappingStack<'parent, 'a> {
    mappings: HashMap<Name, Name>,
    defs: &'a mut HashSet<Name>,
    parent: Option<&'parent NameMappingStack<'parent, 'parent>>
}

fn get_mapping<'a>(stack: &'a NameMappingStack, name: &Name) -> Option<&'a Name> {
    let mut current = stack;
    while let Some(parent) = current.parent {
        if let Some(res) = current.mappings.get(name) {
            return Some(res);
        }
        current = parent;
    }
    return current.mappings.get(name);
}

fn is_new_def(stack: &NameMappingStack, name: &Name) -> bool {
    let mut current = stack;
    while let Some(parent) = current.parent {
        if current.defs.contains(name) {
            return true;
        }
        current = parent;
    }
    return current.defs.contains(name);
}

fn rename_name<F>(name: &mut Name, rename_disjunct: &mut F, name_mappings: &mut NameMappingStack)
    where F: FnMut(Name) -> Name
{
    if let Some(new_name) = get_mapping(name_mappings, name) {
        *name = new_name.clone();
    } else if is_new_def(name_mappings, name) {
        let new_name = rename_disjunct(name.clone());
        name_mappings.mappings.insert(name.clone(), new_name.clone());
        name_mappings.defs.insert(new_name.clone());
        *name = new_name;
    }
}

fn rename_in_block(block: &mut Block, parent_scopes: &NameScopeStack, parent_name_mappings: &NameMappingStack)
{
    let mut defs = HashSet::new();
    let mut child = NameMappingStack {
        parent: Some(parent_name_mappings),
        defs: &mut defs,
        mappings: HashMap::new()
    };
    let mut rename_disjunct = parent_scopes.rename_disjunct();
    for statement in &mut block.statements {
        for name in statement.names_mut() {
            rename_name(name, &mut rename_disjunct, &mut child);
        }
    }
    let scopes = parent_scopes.child_scope(block);
    for statement in &mut block.statements {
        for subblock in statement.subblocks_mut() {
            rename_in_block(subblock, &scopes, &mut child);
        }
    }
}

///
/// Renames all occurences of names in `new_defs` in the given block with new, disjunct names and
/// replaces all occurences of names in `rename_mapping` with the respective values. If 
/// the renaming would cause a new name collision further down the scope stack, the new variable is 
/// renamed, too.
/// 
pub fn fix_name_collisions(block: &mut Block, parent_scopes: &NameScopeStack, new_defs: &mut HashSet<Name>, rename_mapping: HashMap<Name, Name>) {
    debug_assert!(rename_mapping.values().all(|k| new_defs.contains(k)));
    for d in new_defs.iter() {
        assert!(parent_scopes.get(d).is_some());
    }
    let name_mappings = NameMappingStack {
        parent: None,
        defs: new_defs,
        mappings: rename_mapping
    };
    rename_in_block(block, parent_scopes, &name_mappings);
}

#[cfg(test)]
use super::super::lexer::lexer::{fragment_lex, lex_str};
#[cfg(test)]
use super::super::parser::{Parser, TopLevelParser};
#[cfg(test)]
use super::super::analysis::defs_test::*;

#[test]
fn test_fix_name_collisions() {

    let mut actual_types = TypeVec::new();
    let mut body = Block::parse(&mut fragment_lex(
        "
    {
        let result: int = 10;
        while (1) {
            if (result < 5) {
                if (result == 0) {
                    return foo(result,);
                    goto result#1;
                }
                {
                    result = result - 1;
                    goto res;
                }
                @result#1
            }
            @target
        }
    }",
    ), &mut actual_types)
    .unwrap();

    let parent_scopes = NameScopeStack::new(&[][..]);
    let scopes = parent_scopes.child_scope(&EnvironmentBuilder::new()
        .add_test_def("result")
        .add_func_def("foo").add_array_param(PrimitiveType::Int, 0).return_type(PrimitiveType::Int, 0)
        .add_test_def("res")
        .add_test_def("target")
        .destruct().1);

    let mut new_defs = HashSet::new();
    new_defs.insert(Name::l("result"));
    new_defs.insert(Name::l("target"));

    let mut name_mappings = HashMap::new();
    name_mappings.insert(Name::l("res"), Name::l("target"));

    fix_name_collisions(&mut body, &scopes, &mut new_defs, name_mappings);

    let mut expected_types = TypeVec::new();
    let expected = Block::parse(&mut fragment_lex(
        "
    {
        let result#1: int = 10;
        while (1) {
            if (result#1 < 5) {
                if (result#1 == 0) {
                    return foo(result#1,);
                    goto result#2;
                }
                {
                    result#1 = result#1 - 1;
                    goto target;
                }
                @result#2
            }
            @target#1
        }
    }",
    ), &mut expected_types)
    .unwrap();

    assert_ast_frag_eq!(expected, body; expected_types.get_lifetime(), actual_types.get_lifetime() );
}