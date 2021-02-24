use super::super::language::prelude::*;
use super::super::analysis::scope::*;
use super::super::analysis::types::*;
use super::super::analysis::concrete_view_resolution::*;

fn create_monomorphized_instance(
    function: &Function, 
    instantiation: TemplateConcreteViewAssignment, 
    global_scope: &DefinitionScopeStack,
    types: &mut TypeVec
) -> Box<Function> {
    let mut instance = function.deep_copy_ast(types);
    
    for (param, original_param) in instance.params.iter().zip(function.params.iter()) {
        if param.variable_type.deref(types.get_lifetime()).is_view() {
            let original_concrete_view = original_param
                .variable_type.deref(types.get_lifetime()).unwrap_view()
                .get_concrete().unwrap().dyn_clone();
            let concrete_view = instantiation.apply(original_concrete_view);

            instance.identifier.extra_data.push(concrete_view.identifier());

            param.variable_type.deref_mut(&mut types.get_lifetime_mut())
                .unwrap_view_mut().concrete = Some(concrete_view);
        }
    }
    determine_types_in_function(&instance, global_scope, types).internal_error();
    return Box::new(instance);
}

#[cfg(test)]
use super::super::analysis::types::determine_types_in_program;

#[test]
fn test_create_monomorphized_instance() {
    let mut program = Program::parse(&mut lex_str("
    
    fn foo(x: &int[,],) {
        let b: &int[,] = x;
    }
    
    ")).unwrap();

    determine_types_in_program(&mut program).unwrap();

    let function = &program.items[0];
    let view_arguments = TemplateConcreteViewAssignment::reference_views(&function.params, program.types.get_lifetime());
    let global_scope = DefinitionScopeStack::new(&*program.items);
    let instance = create_monomorphized_instance(function, view_arguments, &global_scope, &mut program.types);

    let expected = Program::parse(&mut lex_str("
    
    fn foo(x: &int[,],) {
        let b: &int[,] = x;
    }

    ")).unwrap();
    assert_ast_frag_eq!(
        &expected.items[0], &*instance; expected.lifetime(), program.lifetime()
    );
    assert_eq!(
        &ReferenceView::new() as &dyn ConcreteView, 
        instance.body.unwrap()
            .statements[0]
            .downcast::<LocalVariableDeclaration>().unwrap()
            .declaration
            .variable_type.deref(program.lifetime())
            .unwrap_view()
            .get_concrete().unwrap()
    );
}