use super::code_gen::*;
use super::repr::*;
use super::kernel_analysis::*;
use super::impls::zero_view_repr::*;
use super::impls::index_view_repr::*;
use super::impls::scalar_repr::*;
use super::impls::array_repr::*;
use super::super::language::concrete_views::*;
use super::super::language::prelude::*;
use super::super::language::ast_pfor::*;

use std::collections::HashMap;

fn get_variable_storage<'a>(ty: &Type, on_device: bool) -> Box<dyn VariableStorage> {
    if ty.is_scalar() {
        return Box::new(ScalarRepr::new(ty.clone()));
    } else if ty.as_static().is_some() {
        return Box::new(ArrayRepr::new(ty.clone(), on_device));
    }
    let view = &**ty.as_view().unwrap().concrete_view.as_ref().unwrap();
    if let Some(_) = view.downcast::<ViewZeros>() {
        Box::new(ZeroViewRepr::new(ty.clone()))
    } else if let Some(_) = view.downcast::<ViewIndex>() {
        Box::new(IndexViewRepr::new(ty.clone(), on_device))
    } else if let Some(_) = view.downcast::<ViewReference>() {
        Box::new(IndexViewRepr::new(ty.clone(), on_device))
    } else if let Some(v) = view.downcast::<ViewComposed>() {
        unimplemented!()
    } else {
        panic!("Concrete view {:?} either not supported or not allowed at this point.", view)
    }
}

fn get_default_location(ty: &Type) -> bool {
    !ty.is_scalar()
}

fn generate(program: &Program, generator: &mut dyn CodeGenerator) {
    let (function_infos, kernel_infos) = collect_functions(program).unwrap();
    let mut variable_storages: HashMap<Ptr<dyn SymbolDefinition>, Box<dyn VariableStorage>> = HashMap::new();
    program.for_functions(
        &mut |f, scopes| {
            if f.body.is_none() {
                return Ok(());
            }
            for decl in &f.parameters {
                variable_storages.insert(Ptr::from(decl as &dyn SymbolDefinition), get_variable_storage(&decl.var_type, get_default_location(&decl.var_type)));
            }
            if let Some(return_type) = f.return_type() {
                variable_storages.insert(Ptr::from(f as &dyn SymbolDefinition), get_variable_storage(return_type, get_default_location(return_type)));
            }
            let function_info = function_infos.get(&RefEq::from(f)).unwrap();
            generator.write_function(
                f.parameters.iter()
                    .map(|decl| (variable_storages.get(&RefEq::from(decl as &dyn SymbolDefinition)).as_ref().unwrap().get_out_type(), decl.name.name.clone()))
                    .collect(),
                f.return_type()
                    .map(|_| variable_storages.get(&RefEq::from(f as &dyn SymbolDefinition)).as_ref().unwrap().get_out_type()), 
                Box::new(|mut body_gen| {
                    generate_block(
                        f.body.as_ref().unwrap(), 
                        &mut *body_gen, 
                        scopes,
                        &kernel_infos,
                        &mut variable_storages
                    );
                    return Ok(());
                }),
                function_info.called_from_host, function_info.called_from_device
            )
        }
    ).unwrap();
}

fn generate_block<'a>(
    block: &'a Block, 
    generator: &mut dyn BlockGenerator, 
    scopes: &DefinitionScopeStackConst<'_, 'a>, 
    kernel_infos: &HashMap<Ptr<ParallelFor>, KernelInfo>, 
    variable_storages: &mut HashMap<Ptr<'a, dyn SymbolDefinition>, Box<dyn VariableStorage>>
) {
    block.traverse_proper_children(
        scopes,
        &mut |statement, scopes| {
            generator.write_block(Box::new(|mut inner_gen| {
                generate_statement(statement, &mut *inner_gen, scopes, kernel_infos, variable_storages);
                return Ok(());
            })).unwrap();
            return Ok(());
        }
    ).unwrap();
}

fn generate_statement<'a>(
    statement: &'a dyn Statement, 
    generator: &mut dyn BlockGenerator, 
    scopes: &DefinitionScopeStackConst<'_, 'a>, 
    kernel_infos: &HashMap<Ptr<ParallelFor>, KernelInfo>, 
    variable_storages: &mut HashMap<Ptr<'a, dyn SymbolDefinition>, Box<dyn VariableStorage>>
) {
    if let Some(decl) = statement.downcast::<LocalVariableDeclaration>() {
        let variable_storage = get_variable_storage(&decl.declaration.var_type, get_default_location(&decl.declaration.var_type));
        variable_storage.write_init
        variable_storages.insert(Ptr::from(decl as &dyn SymbolDefinition), variable_storage);
    }
}