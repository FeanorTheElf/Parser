use super::super::language::prelude::*;
use super::super::analysis::scope::*;
use super::super::analysis::symbol::*;
use super::super::analysis::types::*;
use super::CudaContext;

pub struct CudaContextImpl<'a, 'b> {
    device_context: bool,
    kernel_id_counter: &'a mut u32,
    scopes: DefinitionScopeStack<'a, 'b>,
    function: Option<&'b Function>
}

impl<'a, 'b> CudaContextImpl<'a, 'b> {
    pub fn new(global: &'b [Box<Function>], counter: &'a mut u32) -> CudaContextImpl<'a, 'b> {
        CudaContextImpl {
            device_context: false,
            kernel_id_counter: counter,
            scopes: ScopeStack::new(global),
            function: None
        }
    }

    #[cfg(test)]
    pub fn in_test_subscope(&'a mut self, defs: &'b [(Name, Type)]) -> CudaContextImpl<'a, 'b> {
        CudaContextImpl {
            device_context: self.device_context,
            kernel_id_counter: self.kernel_id_counter,
            scopes: self.scopes.child_scope(defs),
            function: None
        }
    }
}

impl<'a, 'b> CudaContext<'a, 'b> for CudaContextImpl<'a, 'b> {
    fn generate_unique_identifier(&mut self) -> u32 {
        *self.kernel_id_counter += 1;
        *self.kernel_id_counter
    }

    fn is_device_context(&self) -> bool {
        self.device_context
    }

    fn set_device(&mut self) {
        self.device_context = true;
    }

    fn set_host(&mut self) {
        self.device_context = false;
    }

    fn set_scope(&mut self, scopes: DefinitionScopeStack<'a, 'b>) {
        self.scopes = scopes
    }

    fn get_scopes(&self) -> &DefinitionScopeStack<'a, 'b> {
        &self.scopes
    }

    fn get_current_function(&self) -> &Function {
        self.function.unwrap()
    }

    fn set_current_function(&mut self, func: &'b Function) {
        self.function = Some(func);
    }
}