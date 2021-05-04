use super::super::language::prelude::*;
use std::collections::{HashSet, HashMap};

///
/// Renames all occurences of names in `new_defs` that are not in `rename_mapping` in the 
/// given statement with new, disjoint names and replaces all occurences of names in `rename_mapping` 
/// with the respective values. If the renaming would cause a new name collision further down 
/// the scope stack, the new variable is renamed, too.
/// 
/// # Uses
/// 
/// The standard use case for this function is to fix potential name collisions, after a new
/// variable was introduced at some point in the ast. In this case, one should ensure that the
/// new name is disjoint from all symbol names that are visible at the point of introduction.
/// 
/// However, this might still collide with formerly valid definitions further down the scope
/// stack. Therefore, call this function on all statements in the block following the point of 
/// introduction, with `new_defs` containing the new name and an empty `rename_mapping`. This
/// will rename the variables in those statements to have new names, and do so recursively until
/// everything is well.
/// 
/// The `rename_mapping` parameter can be used if some fixed names must also be renamed further
/// down the stack. Note that each key in `rename_mapping` must be included in `new_defs`, as it
/// should be renamed. In most cases, this will be empty.
/// 
/// # Note
/// 
/// This function currently might rename too much. In particular, if a variable with name A was
/// renamed to name B in some sublock, then in a subsequent (sibling) subblock a variable with again
/// name B will be renamed, although this is not strictly necessary, as the former variable A is
/// not visible there anymore. 
/// 
pub fn fix_name_collisions(
    statement: &mut dyn Statement, 
    parent_scopes: &DefinitionScopeStackMut, 
    new_defs: &mut HashSet<Name>, 
    mut rename_mapping: HashMap<Name, Name>
) -> Result<(), CompileError> {
    assert!(rename_mapping.keys().all(|k| new_defs.contains(k)));
    // new_defs must already be contained in the scope stack
    for d in new_defs.iter() {
        assert!(parent_scopes.get(d).is_some());
    }

    statement.traverse_preorder_mut(
        parent_scopes,
        &mut |statement, scopes| {
            for name in statement.names_mut() {
                if new_defs.contains(name) {
                    if let Some(mapped_name) = rename_mapping.get(name) {
                        // this name refers to a symbol definition that was already renamed,
                        // so rename the reference accordingly
                        *name = mapped_name.clone();
                    } else {
                        // We have a reference to a previously renamed symbol
                        let new_name = scopes.rename_disjunct(name.clone());
                        rename_mapping.insert(name.clone(), new_name.clone());
                        new_defs.insert(new_name.clone());
                        *name = new_name;
                    }
                }
            }
            return RECURSE;
        }
    )
}

#[test]
fn test_fix_name_collisions() {

    let mut body = Block::parse(&mut fragment_lex("
    {
        let result: int init 10;
        while (1) {
            if (result < 5) {
                if (result == 0) {
                    return foo(result,);
                    goto result#1;
                }
                {
                    result = result - 1;
                    goto target;
                }
                @result#1
            }
            @target
        }
    }"), &mut ParserContext::new()).unwrap();

    let mut parent_scopes = DefinitionScopeStackMut::new();
    let mut defs = [
        testdef("result"),
        testdef("foo"),
        testdef("res"),
        testdef("target")
    ];
    defs.iter_mut().for_each(|d| parent_scopes.register_symbol(d));

    let mut new_defs = HashSet::new();
    new_defs.insert(Name::l("result"));
    new_defs.insert(Name::l("target"));

    let mut name_mappings = HashMap::new();
    name_mappings.insert(Name::l("target"), Name::l("res"));

    fix_name_collisions(&mut body, &parent_scopes, &mut new_defs, name_mappings).unwrap();

    let expected = Block::parse(&mut fragment_lex(
        "
    {
        let result#1: int init 10;
        while (1) {
            if (result#1 < 5) {
                if (result#1 == 0) {
                    return foo(result#1,);
                    goto result#2;
                }
                {
                    result#1 = result#1 - 1;
                    goto res;
                }
                @result#2
            }
            @res
        }
    }",
    ), &mut ParserContext::new()).unwrap();

    assert_ast_frag_eq!(expected, body);
}