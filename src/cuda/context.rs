use super::super::language::prelude::*;
use super::super::analysis::scope::*;
use super::super::util::ref_eq::*;
use super::kernel_data::*;
use super::super::analysis::types::Typed;
use super::super::language::backend::*;

use std::collections::HashMap;

pub trait CudaContext<'stack, 'ast: 'stack> {
    fn is_device_context(&self) -> bool;
    fn set_device(&mut self);
    fn set_host(&mut self);
    fn get_scopes(&self) -> &DefinitionScopeStack<'stack, 'ast>;
    fn get_current_function(&self) -> &Function;
    fn set_current_function(&mut self, function: &'ast Function);

    fn get_scopes_mut_do_not_use_outside_of_cuda_context(&mut self) -> &mut DefinitionScopeStack<'stack, 'ast>;

    fn calculate_type(&self, expr: &Expression) -> Type {
        expr.calculate_type(self.get_scopes())
    }

}

impl<'stack, 'ast: 'stack> dyn CudaContext<'stack, 'ast> + '_ {
    
    pub fn enter_scope<S: ?Sized>(&mut self, scope: &'ast S)
    where
        &'ast S: EnumerateDefinitions<'ast>
    {
        self.get_scopes_mut_do_not_use_outside_of_cuda_context().enter(scope)
    }

    pub fn exit_scope(&mut self) {
        self.get_scopes_mut_do_not_use_outside_of_cuda_context().exit()
    }

    pub fn calculate_var_type<'a>(&'a self, variable: &'a Name, pos: &'a TextPosition) -> Type {
        self.get_scopes().get_defined(variable, pos).unwrap().calc_type()
    }
}

impl<'stack, 'ast: 'stack, T: CudaContext<'stack, 'ast>> CudaContext<'stack, 'ast> for Box<T> {
    fn is_device_context(&self) -> bool {
        (**self).is_device_context()
    }

    fn set_device(&mut self) {
        (**self).set_device()
    }

    fn set_host(&mut self) {
        (**self).set_host()
    }

    fn get_scopes(&self) -> &DefinitionScopeStack<'stack, 'ast> {
        (**self).get_scopes()
    }

    fn calculate_type(&self, expr: &Expression) -> Type {
        (**self).calculate_type(expr)
    }

    fn get_current_function(&self) -> &Function {
        (**self).get_current_function()
    }

    fn set_current_function(&mut self, function: &'ast Function) {
        (**self).set_current_function(function)
    }

    fn get_scopes_mut_do_not_use_outside_of_cuda_context(&mut self) -> &mut DefinitionScopeStack<'stack, 'ast> {
        (**self).get_scopes_mut_do_not_use_outside_of_cuda_context()
    }
}

pub struct CudaContextImpl<'b, 'ast> {
    device_context: bool,
    scopes: DefinitionScopeStack<'b, 'ast>,
    function: Option<&'ast Function>,
    functions: HashMap<Ref<'ast, Function>, FunctionInfo<'ast>>,
    kernels: HashMap<Ref<'ast, ParallelFor>, KernelInfo<'ast>>
}

impl<'b, 'ast> CudaContextImpl<'b, 'ast> {
    pub fn build_for_program(global: &'ast Program) -> Result<CudaContextImpl<'b, 'ast>, OutputError> {
        let mut counter: u32 = 0;
        let (functions, kernels) = collect_functions(global, &mut || { counter += 1; counter })?;
        Ok(CudaContextImpl {
            device_context: false,
            scopes: ScopeStack::new(&*global.items),
            function: None,
            functions: functions,
            kernels: kernels
        })
    }
}

impl<'b, 'ast: 'b> CudaContext<'b, 'ast> for CudaContextImpl<'b, 'ast> {

    fn is_device_context(&self) -> bool {
        self.device_context
    }

    fn set_device(&mut self) {
        self.device_context = true;
    }

    fn set_host(&mut self) {
        self.device_context = false;
    }

    fn get_scopes(&self) -> &DefinitionScopeStack<'b, 'ast> {
        &self.scopes
    }

    fn get_current_function(&self) -> &Function {
        self.function.unwrap()
    }

    fn set_current_function(&mut self, func: &'ast Function) {
        self.function = Some(func);
    }
    
    fn get_scopes_mut_do_not_use_outside_of_cuda_context(&mut self) -> &mut DefinitionScopeStack<'b, 'ast> {
        &mut self.scopes
    }
}