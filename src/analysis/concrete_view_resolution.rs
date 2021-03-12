use super::super::language::prelude::*;
use super::scope::DefinitionScopeStack;
use super::export::*;
use super::types::*;

use super::super::util::ref_eq::Ptr;
use super::topological_sort::call_graph_topological_sort;

use std::collections::{HashSet, HashMap};

fn for_each_function_call<'a, 'b, F>(
    function: &'a Function, 
    global_scope: &DefinitionScopeStack<'b, 'a>, 
    mut f: F
) -> Result<(), CompileError>
    where F: for<'c> FnMut(&'a FunctionCall, &DefinitionScopeStack<'c, 'a>) -> Result<(), CompileError>
{
    if let Some(body) = &function.body {
        global_scope.child_scope(function).try_scoped_preorder_depth_first_search(body, &mut |block, scope| {
            for statement in &block.statements {
                for expression in statement.expressions() {
                    expression.try_call_tree_preorder_depth_first_search(&mut |call| f(call, scope))?;
                }
            }
            return Ok(());
        })?;
    }
    return Ok(());
}

#[derive(Hash, PartialEq, Eq, Debug)]
pub struct TemplateConcreteViewAssignment {
    mapping: std::collections::BTreeMap<Template, Box<dyn ConcreteView>>
}

impl TemplateConcreteViewAssignment {
    ///
    /// Creates an assignment in which each template view is mapped to
    /// a reference view
    /// 
    pub fn reference_views(
        parameters: &Vec<Declaration>, 
        type_lifetime: Lifetime
    ) -> TemplateConcreteViewAssignment {
        let mapping = parameters.iter()
            .map(|p| type_lifetime.cast(p.variable_type))
            .filter_map(|t| t.expect_view(&TextPosition::NONEXISTING).ok())
            .map(|t| t.get_concrete().unwrap())
            .filter_map(|t| t.downcast::<Template>())
            .map(|t| (*t, Box::new(ReferenceView::new()) as Box<dyn ConcreteView>))
            .collect();
        return TemplateConcreteViewAssignment { mapping };
    }

    ///
    /// Creates a new assignment that assigns the i-th given view to the i-th 
    /// given template view parameter. If the given views contain templates themselves,
    /// they are replaced using the current object mapping. Therefore, this function
    /// can be used to determine the concrete view instantiation for a function called
    /// from a function for which a concrete instantiation is given. 
    /// 
    /// Panics if a concrete view contains a template that is not mapped to a
    /// concrete view by this mapping.
    /// 
    fn with_call_views(&self, 
        parameters: &Vec<Declaration>, 
        views: &Vec<Box<dyn ConcreteView>>, 
        type_lifetime: Lifetime
    ) -> TemplateConcreteViewAssignment {
        let mapping = parameters.iter()
            .map(|p| p.variable_type.deref(type_lifetime))
            .filter_map(|t| t.expect_view(&TextPosition::NONEXISTING).ok())
            .map(|t| t.get_concrete().unwrap())
            .filter_map(|t| t.downcast::<Template>())
            .zip(views)
            .map(|(template, concrete)| {
                return (*template, self.apply_complete(concrete.clone()));
            }).collect();
        return TemplateConcreteViewAssignment { mapping };
    }

    ///
    /// Replaces all template occurences in `view` that are keys in this mapping
    /// by the corresponding mapped concrete view
    /// 
    pub fn apply(&self, view: Box<dyn ConcreteView>) -> Box<dyn ConcreteView> {
        self.mapping.iter()
            .fold(view, |current, (template, target)| 
                current.replace_templated(*template, &**target)
            )
    }
    
    ///
    /// Replaces all template occurences in `view` that are keys in this mapping
    /// by the corresponding mapped concrete view (same as `apply`)
    /// 
    /// This function panics if not all template occurences in view are successfully
    /// replaced by template-free concrete views
    /// 
    pub fn apply_complete(&self, view: Box<dyn ConcreteView>) -> Box<dyn ConcreteView> {
        let result = self.apply(view);
        assert!(!result.contains_templated());
        return result;
    }
}

///
/// Calculates the concrete view types that are passed to called function in the given 
/// function call. The returned vector contains exactly one concrete view descriptor 
/// for each templated view parameter in the called function, that describes the 
/// concrete view type of the argument passed to that parameter.
/// 
fn calculated_function_call_concrete_view_arguments(
    call: &FunctionCall, 
    called_function: &Function, 
    scope: &DefinitionScopeStack, 
    types: &TypeVec
) -> Result<Vec<Box<dyn ConcreteView>>, CompileError> {
    let mut call_types = Vec::new();
    for (formal_param, given_param) in called_function.params.iter().zip(call.parameters.iter()) {
        if formal_param.variable_type.deref(types.get_lifetime()).is_view() {
            let given_type_ptr = get_expression_type(given_param, scope).unwrap();
            let given_type = given_type_ptr.deref(types.get_lifetime());
            let expected_type = formal_param.variable_type.deref(types.get_lifetime());
            debug_assert!(given_type.is_implicitly_convertable(expected_type, types.get_lifetime()));
            
            call_types.push(
                given_type.expect_view(given_param.pos()).internal_error()
                    .get_concrete().unwrap().dyn_clone()
            );
        }
    }
    return Ok(call_types);
}

pub type RequiredInstantiationsMap<'a> = HashMap<
    Ptr<'a, Function>, HashSet<TemplateConcreteViewAssignment>
>;

pub fn calculate_required_function_instantiations<'a>(
    functions: &'a [Box<Function>], 
    types: &mut TypeVec
) -> Result<RequiredInstantiationsMap<'a>, CompileError> {

    let mut instantiations: RequiredInstantiationsMap = functions.iter()
        .map(|function| (Ptr::from(&**function), HashSet::new()))
        .collect();
    
    for exported_function in get_functions_to_export(functions) {
        let function_instantiations = instantiations.get_mut(&Ptr::from(exported_function)).unwrap();
        function_instantiations.insert(TemplateConcreteViewAssignment::reference_views(
            &exported_function.params, types.get_lifetime()
        ));
    }
    
    let global_scope = DefinitionScopeStack::new(functions);
    for function in call_graph_topological_sort(functions)? {
        for_each_function_call(function, &global_scope, |call, scope| {
            let called_function_identifier = &call.function.expect_identifier()?.identifier;
            if let Identifier::Name(called_function_name) = called_function_identifier {
                
                let called_function = scope
                    .get_defined(called_function_name, call.pos())?
                    .dynamic().downcast_ref::<Function>().unwrap();
                let concrete_view_arguments = calculated_function_call_concrete_view_arguments(
                    call, called_function, scope, types
                )?;

                let mut new_instantiations = HashSet::new();
                for instantiation in instantiations.get(&Ptr::from(function)).unwrap() {
                    new_instantiations.insert(instantiation.with_call_views(
                        &called_function.params, &concrete_view_arguments, types.get_lifetime())
                    );
                }

                let called_function_instantiations = instantiations
                    .entry(Ptr::from(called_function))
                    .or_insert_with(HashSet::new);
                called_function_instantiations.extend(new_instantiations.into_iter());
            }
            return Ok(());
        })?;
    }

    return Ok(instantiations);
}

#[test]
fn test_calculate_required_function_instantiations() {
    let mut program = Program::parse(&mut lex_str("
    
    fn main(a: &int[,],) {
        foo(a,);
        foo(zeros(10,),);
    }

    fn foo(x: &int[,],) {

    }
    
    ")).unwrap();

    determine_types_in_program(&mut program).unwrap();

    let required_instantiations = calculate_required_function_instantiations(
        &program.items, &mut program.types
    ).unwrap();

    let main_instantiation = TemplateConcreteViewAssignment {
        mapping: vec![
            (Template::new(0), Box::new(ReferenceView::new()) as Box<dyn ConcreteView>)
        ].into_iter().collect()
    };
    let foo_ref_instantiation = TemplateConcreteViewAssignment {
        mapping: vec![
            (Template::new(0), Box::new(ReferenceView::new()) as Box<dyn ConcreteView>)
        ].into_iter().collect()
    };
    let foo_zero_instantiation = TemplateConcreteViewAssignment {
        mapping: vec![
            (Template::new(0), Box::new(ZeroView::new()) as Box<dyn ConcreteView>)
        ].into_iter().collect()
    };
    let expected: HashMap<_, _> = vec![
        (Ptr::from(program.get_function("main")), vec![main_instantiation].into_iter().collect()), 
        (Ptr::from(program.get_function("foo")), vec![foo_ref_instantiation, foo_zero_instantiation].into_iter().collect())
    ].into_iter().collect();
    assert_eq!(expected, required_instantiations);
}