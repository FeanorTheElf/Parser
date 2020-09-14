use super::super::analysis::scope::*;
use super::super::analysis::types::Typed;
use super::super::language::prelude::*;
use super::super::util::ref_eq::*;
use super::kernel_data::*;

use std::collections::HashMap;

pub trait CudaContext<'data, 'ast: 'data> {
    fn is_device_context(&self) -> bool;

    fn set_device(&mut self);

    fn set_host(&mut self);

    fn ast_lifetime(&self) -> Lifetime<'ast>;

    fn get_scopes(&self) -> &DefinitionScopeStack<'data, 'ast>;

    fn get_current_function(&self) -> &Function;

    fn set_current_function(&mut self, function: &'ast Function);

    fn get_pfor_data(&self, pfor: &ParallelFor) -> &'data KernelInfo<'ast>;

    fn get_function_data(&self, function: &Function) -> &'data FunctionInfo<'ast>;

    fn get_scopes_mut_do_not_use_outside_of_cuda_context(
        &mut self,
    ) -> &mut DefinitionScopeStack<'data, 'ast>;

    fn calculate_type(&self, expr: &Expression) -> Type {

        expr.calculate_type(self.get_scopes(), self.ast_lifetime()).internal_error()
    }
}

impl<'data, 'ast: 'data> dyn CudaContext<'data, 'ast> + '_ {
    pub fn enter_scope<'b, S: ?Sized>(&mut self, scope: &'b S)
    where
        &'b S: EnumerateDefinitions<'ast>,
    {

        self.get_scopes_mut_do_not_use_outside_of_cuda_context()
            .enter(scope)
    }

    pub fn exit_scope(&mut self) {

        self.get_scopes_mut_do_not_use_outside_of_cuda_context()
            .exit()
    }

    pub fn calculate_var_type<'a>(&'a self, variable: &'a Name, pos: &'a TextPosition) -> Ref<'a, Type> {
        self.ast_lifetime().cast(self.get_scopes().get_defined(variable, pos).unwrap().get_type()).borrow()
    }
}

impl<'data, 'ast: 'data, T: CudaContext<'data, 'ast>> CudaContext<'data, 'ast> for Box<T> {
    fn is_device_context(&self) -> bool {

        (**self).is_device_context()
    }

    fn get_pfor_data(&self, pfor: &ParallelFor) -> &'data KernelInfo<'ast> {

        (**self).get_pfor_data(pfor)
    }

    fn get_function_data(&self, function: &Function) -> &'data FunctionInfo<'ast> {

        (**self).get_function_data(function)
    }

    fn set_device(&mut self) {

        (**self).set_device()
    }

    fn set_host(&mut self) {

        (**self).set_host()
    }

    fn get_scopes(&self) -> &DefinitionScopeStack<'data, 'ast> {

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

    fn get_scopes_mut_do_not_use_outside_of_cuda_context(
        &mut self,
    ) -> &mut DefinitionScopeStack<'data, 'ast> {

        (**self).get_scopes_mut_do_not_use_outside_of_cuda_context()
    }

    fn ast_lifetime(
        &self,
    ) -> Lifetime<'ast> {

        (**self).ast_lifetime()
    }
}

pub struct CudaContextImpl<'data, 'ast> {
    device_context: bool,
    scopes: DefinitionScopeStack<'data, 'ast>,
    function: Option<&'ast Function>,
    ast_lifetime: Lifetime<'ast>,
    functions: &'data HashMap<Ptr<'ast, Function>, FunctionInfo<'ast>>,
    kernels: &'data HashMap<Ptr<'ast, ParallelFor>, KernelInfo<'ast>>,
}

impl<'data, 'ast: 'data> CudaContextImpl<'data, 'ast> {
    pub fn new(
        global: &'ast Program,
        functions: &'data HashMap<Ptr<'ast, Function>, FunctionInfo<'ast>>,
        kernels: &'data HashMap<Ptr<'ast, ParallelFor>, KernelInfo<'ast>>,
    ) -> CudaContextImpl<'data, 'ast> {

        CudaContextImpl {
            device_context: false,
            scopes: ScopeStack::new(&*global.items),
            function: None,
            functions: functions,
            kernels: kernels,
            ast_lifetime: global.types.get_lifetime()
        }
    }
}

impl<'data, 'ast: 'data> CudaContext<'data, 'ast> for CudaContextImpl<'data, 'ast> {
    fn is_device_context(&self) -> bool {

        self.device_context
    }

    fn get_pfor_data(&self, pfor: &ParallelFor) -> &'data KernelInfo<'ast> {

        self.kernels.get(&RefEq::from(pfor)).unwrap()
    }

    fn get_function_data(&self, function: &Function) -> &'data FunctionInfo<'ast> {

        self.functions.get(&RefEq::from(function)).unwrap()
    }

    fn set_device(&mut self) {

        self.device_context = true;
    }

    fn set_host(&mut self) {

        self.device_context = false;
    }

    fn get_scopes(&self) -> &DefinitionScopeStack<'data, 'ast> {

        &self.scopes
    }

    fn get_current_function(&self) -> &Function {

        self.function.unwrap()
    }

    fn set_current_function(&mut self, func: &'ast Function) {

        self.function = Some(func);
    }

    fn get_scopes_mut_do_not_use_outside_of_cuda_context(
        &mut self,
    ) -> &mut DefinitionScopeStack<'data, 'ast> {

        &mut self.scopes
    }

    fn ast_lifetime(&self) -> Lifetime<'ast> {
        self.ast_lifetime
    }
}
