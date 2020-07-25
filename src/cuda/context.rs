use super::super::language::prelude::*;
use super::super::analysis::scope::*;
use super::super::analysis::symbol::*;
use super::super::analysis::types::*;
use super::CudaContext;

pub struct CudaContextImpl<'a, 'b> {
    device_context: bool,
    kernel_id_counter: &'a mut u32,
    scopes: DefinitionScopeStack<'a, 'b>
}

impl<'a, 'b> CudaContextImpl<'a, 'b> {
    pub fn new(global: &'b [Box<Function>], counter: &'a mut u32) -> CudaContextImpl<'a, 'b> {
        CudaContextImpl {
            device_context: false,
            kernel_id_counter: counter,
            scopes: ScopeStack::new(global)
        }
    }

    #[cfg(test)]
    pub fn in_test_subscope(&'a mut self, defs: &'b [(Name, Type)]) -> CudaContextImpl<'a, 'b> {
        CudaContextImpl {
            device_context: true,
            kernel_id_counter: self.kernel_id_counter,
            scopes: self.scopes.child_scope(defs)
        }
    }
}

impl<'a, 'b> CudaContext for CudaContextImpl<'a, 'b> {
    fn generate_unique_identifier(&mut self) -> u32 {
        *self.kernel_id_counter += 1;
        *self.kernel_id_counter
    }

    fn is_device_context(&self) -> bool {
        self.device_context
    }

    fn in_device<'c>(&'c mut self) -> Box<dyn 'c + CudaContext> {
        Box::new(CudaContextImpl {
            device_context: true,
            kernel_id_counter: self.kernel_id_counter,
            scopes: self.scopes.child_stack()
        })
    }

    fn in_scope<'c>(&'c mut self, scopes: ScopeStack<'c, &'c dyn SymbolDefinition>) -> Box<dyn 'c + CudaContext> {
        Box::new(CudaContextImpl {
            device_context: true,
            kernel_id_counter: self.kernel_id_counter,
            scopes: scopes
        })
    }

    fn get_scopes(&self) -> &ScopeStack<&dyn SymbolDefinition> {
        &self.scopes
    }
}