use super::super::language::prelude::*;
use super::super::analysis::concrete_view::*;

fn create_monomorphized_instance(function: &Function, instantiation: TemplateConcreteViewAssignment, types: &mut TypeVec) -> Box<Function> {
    let mut instance = function.clone(types);
    unimplemented!()
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

    let instance = create_monomorphized_instance(function, view_arguments, &mut program.types);
}