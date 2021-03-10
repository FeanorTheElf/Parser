use super::super::language::prelude::*;
use super::super::analysis::scope::*;
use super::super::analysis::types::*;
use super::super::analysis::concrete_view_resolution::*;

fn get_concrete_view_mangling_parts<'a>(
    function: &'a Function, 
    instantiation: &'a TemplateConcreteViewAssignment, 
    type_lifetime: Lifetime<'a>
) -> impl 'a + Iterator<Item = String> {

    function.params.iter()
        .filter_map(move |p| 
            p.variable_type.deref(type_lifetime).expect_view(&TextPosition::NONEXISTING).ok())
        .map(|v| v.get_concrete().unwrap())
        .inspect(|v| debug_assert!(v.any().is::<Template>()))
        .map(move |v| instantiation.apply_complete(v.dyn_clone()))
        .map(|v| v.identifier())
}

fn create_monomorphized_instance(
    function: &Function, 
    instantiation: TemplateConcreteViewAssignment, 
    global_scope: &DefinitionScopeStack,
    types: &mut TypeVec
) -> Box<Function> {
    let mut instance = function.deep_copy_ast(types);

    instance.identifier.extra_data.extend(
        get_concrete_view_mangling_parts(function, &instantiation, types.get_lifetime())
    );
    
    for (param, original_param) in instance.params.iter().zip(function.params.iter()) {
        if param.variable_type.deref(types.get_lifetime()).is_view() {
            let original_concrete_view = original_param
                .variable_type.deref(types.get_lifetime()).unwrap_view()
                .get_concrete().unwrap().dyn_clone();
            let concrete_view = instantiation.apply_complete(original_concrete_view);

            param.variable_type.deref_mut(&mut types.get_lifetime_mut())
                .unwrap_view_mut().concrete = Some(concrete_view);
        }
    }
    determine_types_in_function(&instance, global_scope, types).internal_error();
    return Box::new(instance);
}

fn monomorphize(
    program: &mut Program
) -> Result<(), CompileError> {
    let required_instantiations = calculate_required_function_instantiations(
        &program.items, &mut program.types
    )?;
    let global_scope = DefinitionScopeStack::new(&*program.items);
    let types: &mut TypeVec = &mut program.types;
    let functions = required_instantiations.into_iter()
        .flat_map(|(function, instances_params)| 
            instances_params.into_iter().map(move |instance_params| (function, instance_params))
        )
        .map(|(function, instance_params)| 
            create_monomorphized_instance(
                &*function, instance_params, &global_scope, types
            )
        )
        .collect::<Vec<_>>();
    program.items = functions;
    return Ok(());
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
    let view_arguments = TemplateConcreteViewAssignment::reference_views(
        &function.params, program.types.get_lifetime()
    );
    let global_scope = DefinitionScopeStack::new(&*program.items);
    let instance = create_monomorphized_instance(
        function, view_arguments, &global_scope, &mut program.types
    );

    let expected = Program::parse(&mut lex_str("
    
    fn foo__r(x: &int[,],) {
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