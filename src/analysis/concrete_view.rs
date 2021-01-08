use super::super::language::prelude::*;
use super::super::language::position::NONEXISTING;
use super::scope::DefinitionScopeStack;
use super::export::*;
use super::types::*;
use super::type_error::*;

use super::super::util::ref_eq::Ptr;
use super::topological_sort::call_graph_topological_sort;

use std::collections::{HashSet, HashMap};

fn for_each_named_function_call<F>(function: &Function, global_scope: &DefinitionScopeStack, mut f: F) -> Result<(), CompileError>
    where F: FnMut(&FunctionCall, &DefinitionScopeStack) -> Result<(), CompileError>
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
    fn reference_views(parameters: &Vec<Declaration>, type_lifetime: Lifetime) -> TemplateConcreteViewAssignment {
        let mapping = parameters.iter().map(|p| type_lifetime.cast(p.variable_type))
            .filter_map(|t| t.expect_view(&NONEXISTING).ok())
            .map(|t| t.get_concrete().unwrap())
            .filter_map(|t| t.downcast::<Template>())
            .map(|t| (*t, Box::new(ReferenceView::new()) as Box<dyn ConcreteView>)).collect();
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
    fn with_call_views(&self, parameters: &Vec<Declaration>, views: &Vec<Box<dyn ConcreteView>>, type_lifetime: Lifetime) -> TemplateConcreteViewAssignment {
        let mapping = parameters.iter().map(|p| p.variable_type.deref(type_lifetime))
            .filter_map(|t| t.expect_view(&NONEXISTING).ok())
            .map(|t| t.get_concrete().unwrap())
            .filter_map(|t| t.downcast::<Template>())
            .zip(views)
            .map(|(template, concrete)| {
                let target = self.apply(concrete.clone());
                assert!(!target.contains_templated());
                return (*template, target);
            }).collect();
        return TemplateConcreteViewAssignment { mapping };
    }

    pub fn apply(&self, view: Box<dyn ConcreteView>) -> Box<dyn ConcreteView> {
        self.mapping.iter().fold(view, |current, (template, target)| current.replace_templated(*template, &**target))
    }
}

///
/// Calculates the concrete view types that are passed to called function in the given 
/// function call. The returned vector contains exactly one concrete view descriptor for
/// each templated view parameter in the called function, that describes the concrete view
/// type of the argument passed to that parameter.
/// 
fn calculated_function_call_concrete_view_arguments(call: &FunctionCall, called_function: &Function, scope: &DefinitionScopeStack, types: &TypeVec) -> Result<Vec<Box<dyn ConcreteView>>, CompileError> {
    let mut call_types = Vec::new();
    for (formal_param, given_param) in called_function.params.iter().zip(call.parameters.iter()) {
        if formal_param.variable_type.deref(types.get_lifetime()).is_view() {
            let given_type_ptr = get_expression_type(given_param, scope).unwrap();
            let given_type = given_type_ptr.deref(types.get_lifetime());
            let expected_type = formal_param.variable_type.deref(types.get_lifetime());
            assert!(given_type.is_implicitly_convertable(expected_type, types.get_lifetime()));
            call_types.push(given_type.expect_view(given_param.pos()).internal_error().get_concrete().unwrap().dyn_clone());
        }
    }
    return Ok(call_types);
}

pub type RequiredInstantiationsMap<'a> = HashMap<Ptr<'a, Function>, HashSet<TemplateConcreteViewAssignment>>;

pub fn calculate_required_function_instantiations<'a>(functions: &'a [Box<Function>], types: &mut TypeVec) -> Result<RequiredInstantiationsMap<'a>, CompileError> {

    let mut instantiations: RequiredInstantiationsMap = functions.iter().map(|function| (Ptr::from(&**function), HashSet::new())).collect();
    for exported_function in get_functions_to_export(functions) {
        let function_data = instantiations.get_mut(&Ptr::from(exported_function)).unwrap();
        function_data.insert(TemplateConcreteViewAssignment::reference_views(&exported_function.params, types.get_lifetime()));
    }
    
    let global_scope = DefinitionScopeStack::new(functions);
    for function in call_graph_topological_sort(functions)? {

        for_each_named_function_call(function, &global_scope, |call, scope| {
            if let Identifier::Name(called_function_name) = &call.function.expect_identifier()?.identifier {
                let called_function = scope.get_defined(called_function_name, call.pos())?.dynamic().downcast_ref::<Function>().unwrap();
                let concrete_view_arguments = calculated_function_call_concrete_view_arguments(call, called_function, scope, types)?;
                let mut new_instantiations = Vec::new();
                for instantiation in instantiations.get(&Ptr::from(function)).unwrap() {
                    new_instantiations.push(instantiation.with_call_views(&called_function.params, &concrete_view_arguments, types.get_lifetime()));
                }
            }
            return Ok(());
        })?;
    }

    return Ok(instantiations);
}

fn set_concrete_type_box(target: TypePtr, concrete: Box<dyn ConcreteView>, type_lifetime: &mut LifetimeMut) {
    let target_mut = target.deref_mut(type_lifetime);
    if let Type::View(view) = target_mut {
        assert!(view.concrete.is_none());
        view.concrete = Some(concrete);
    } else {
        panic!("Cannot set concrete type of non-view type");
    }
}

fn set_concrete_type<T: ConcreteView>(target: TypePtr, concrete: T, type_lifetime: &mut LifetimeMut) {
    set_concrete_type_box(target, Box::new(concrete), type_lifetime);
}

#[derive(Debug)]
struct NotAllRequiredConcreteTypesKnown {
    view_with_unknown_concrete_type: TypePtr
}

fn set_builtin_call_result_concrete_type(call: &FunctionCall, op: BuiltInIdentifier, scopes: &DefinitionScopeStack, type_lifetime: &mut LifetimeMut) -> Result<(), NotAllRequiredConcreteTypesKnown>
{
    assert!(call.function == op);
    assert!(call.result_type_cache.get().is_some());

    match op {
        BuiltInIdentifier::ViewZeros => {
            assert!(call.get_stored_type().deref(type_lifetime.as_const()).is_view());
            set_concrete_type(call.get_stored_type(), ZeroView::new(), type_lifetime);
        },
        BuiltInIdentifier::FunctionAdd |
            BuiltInIdentifier::FunctionMul |
            BuiltInIdentifier::FunctionAnd |
            BuiltInIdentifier::FunctionOr |
            BuiltInIdentifier::FunctionEq |
            BuiltInIdentifier::FunctionGeq |
            BuiltInIdentifier::FunctionLeq |
            BuiltInIdentifier::FunctionNeq |
            BuiltInIdentifier::FunctionLs |
            BuiltInIdentifier::FunctionGt |
            BuiltInIdentifier::FunctionUnaryDiv | 
            BuiltInIdentifier::FunctionUnaryNeg => {
                assert!(!call.get_stored_type().deref(type_lifetime.as_const()).is_view());
            },
        BuiltInIdentifier::FunctionIndex => {
            assert!(call.get_stored_type().deref(type_lifetime.as_const()).is_view());
            let indexed_type = get_expression_type(&call.parameters[0], scopes).unwrap();
            let concrete: Box<dyn ConcreteView> = match indexed_type.deref(type_lifetime.as_const()) {
                Type::View(view) => {
                    debug_assert!(view.base.dimension == call.parameters.len() - 1);
                    if let Some(indexed_concrete_view) = &view.concrete {
                        Box::new(ComposedView::compose(Box::new(IndexView::new(view.base.dimension)), indexed_concrete_view.dyn_clone()))
                    } else {
                        return Err(NotAllRequiredConcreteTypesKnown {
                            view_with_unknown_concrete_type: indexed_type
                        });
                    }
                },
                Type::Array(arr) => {
                    debug_assert!(arr.dimension == call.parameters.len() - 1);
                    Box::new(IndexView::new(arr.dimension))
                },
                _ => unimplemented!()
            };
            set_concrete_type_box(call.get_stored_type(), concrete, type_lifetime);
        }
    };
    return Ok(());
}

fn set_expression_concrete_type<'a>(expr: &'a Expression, scopes: &DefinitionScopeStack, type_lifetime: &mut LifetimeMut) -> Result<(), NotAllRequiredConcreteTypesKnown> {
    let mut result = Ok(());
    match expr {
        Expression::Call(call) => {
            result = result.and(set_expression_concrete_type(&call.function, scopes, type_lifetime));
            for param in &call.parameters {
                result = result.and(set_expression_concrete_type(param, scopes, type_lifetime));
            }
            if let Expression::Variable(var) = &call.function {
                if let Identifier::BuiltIn(op) = &var.identifier {
                    result = result.and(set_builtin_call_result_concrete_type(&**call, *op, scopes, type_lifetime));
                }
            }
        },
        Expression::Variable(_) => {},
        Expression::Literal(_) => {}
    };
    return result;
}

fn determine_concrete_view_types_block(block: &Block, parent_scopes: &DefinitionScopeStack, type_lifetime: &mut LifetimeMut) -> Result<(), CompileError> {

    let scopes = parent_scopes.child_scope(block);
    for statement in &block.statements {
        for expression in statement.expressions() {
            // TODO: handle cyclic type dependencies correctly
            set_expression_concrete_type(expression, &scopes, type_lifetime).unwrap();
        }
        if let Some(declaration) = statement.downcast::<LocalVariableDeclaration>() {
            let is_var_type_view = declaration.declaration.variable_type.deref(type_lifetime.as_const()).is_view();
            if let Some(value) = &declaration.value {
                if is_var_type_view {
                    let concrete = if let Type::View(value_view) = get_expression_type(value, &scopes).unwrap().deref(type_lifetime.as_const()) {
                        value_view.concrete.as_ref().unwrap().dyn_clone()
                    } else {
                        return Err(error_type_not_convertable(declaration.pos(), get_expression_type(value, &scopes).unwrap(), declaration.declaration.variable_type, type_lifetime.as_const()))
                    };
                    set_concrete_type_box(declaration.declaration.variable_type, concrete, type_lifetime);
                }
            } else if is_var_type_view {
                return Err(error_view_not_initialized(declaration, type_lifetime.as_const()));
            }
        }
    }

    return Ok(());
}

pub fn determine_concrete_view_types(functions: &[Box<Function>], type_lifetime: &mut LifetimeMut) -> Result<(), CompileError> {

    let scopes = DefinitionScopeStack::new(functions);
    for function in functions {
        let mut template_id = 0;
        for param in &function.params {
            if let Type::View(view) = param.variable_type.deref_mut(type_lifetime) {
                view.concrete = Some(Box::new(Template::new(template_id)));
            }
            template_id += 1;
        }

        let function_scope = scopes.child_scope(&**function);
        if let Some(body) = &function.body {
            determine_concrete_view_types_block(body, &function_scope, type_lifetime)?;
        }
    }

    return Ok(());
}

#[cfg(test)]
use super::super::language::ast_test::*;

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

    determine_concrete_view_types(&program.items, &mut program.types.get_lifetime_mut()).unwrap();

    let required_instantiations = calculate_required_function_instantiations(&program.items, &mut program.types).unwrap();

    let main_instantiation = TemplateConcreteViewAssignment {
        mapping: vec![(Template::new(0), Box::new(ReferenceView::new()) as Box<dyn ConcreteView>)].into_iter().collect()
    };
    let foo_ref_instantiation = TemplateConcreteViewAssignment {
        mapping: vec![(Template::new(0), Box::new(ReferenceView::new()) as Box<dyn ConcreteView>)].into_iter().collect()
    };
    let foo_zero_instantiation = TemplateConcreteViewAssignment {
        mapping: vec![(Template::new(0), Box::new(ZeroView::new()) as Box<dyn ConcreteView>)].into_iter().collect()
    };
    let expected: HashMap<_, _> = vec![(Ptr::from(program.get_function("main")), vec![main_instantiation].into_iter().collect()), 
        (Ptr::from(program.get_function("foo")), vec![foo_ref_instantiation, foo_zero_instantiation].into_iter().collect())].into_iter().collect();
    assert_eq!(expected, required_instantiations);
}