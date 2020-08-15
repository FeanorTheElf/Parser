use super::language::prelude::*;
use super::analysis::scope::*;
use super::analysis::symbol::*;
use super::analysis::types::*;

pub mod writer;

pub trait CudaContext {
    fn generate_unique_identifier(&mut self) -> u32;
    fn is_device_context(&self) -> bool;
    fn in_device<'a>(&'a mut self) -> Box<dyn 'a + CudaContext>;
    fn in_scope<'a>(&'a mut self, scopes: DefinitionScopeStack<'a, 'a>) -> Box<dyn 'a + CudaContext>;
    fn get_scopes(&self) -> &DefinitionScopeStack;

    fn calculate_type(&self, expr: &Expression) -> Type {
        expr.calculate_type(self.get_scopes())
    }

    fn calculate_var_type(&self, variable: &Name) -> Type {
        self.get_scopes().get(variable).expect(format!("Expected {} to be defined", variable).as_str()).calc_type()
    }
}

impl<T: CudaContext> CudaContext for Box<T> {
    fn generate_unique_identifier(&mut self) -> u32 {
        self.generate_unique_identifier()
    }

    fn is_device_context(&self) -> bool {
        self.is_device_context()
    }

    fn in_device<'a>(&'a mut self) -> Box<dyn 'a + CudaContext> {
        self.in_device()
    }
    
    fn in_scope<'a>(&'a mut self, scopes: ScopeStack<'a, &'a dyn SymbolDefinition>) -> Box<dyn 'a + CudaContext> {
        self.in_scope(scopes)
    }

    fn get_scopes(&self) -> &ScopeStack<&dyn SymbolDefinition> {
        self.get_scopes()
    }

    fn calculate_var_type(&self, variable: &Name) -> Type {
        self.calculate_var_type(variable)
    }

    fn calculate_type(&self, expr: &Expression) -> Type {
        self.calculate_type(expr)
    }
}

const INDEX_TYPE: &'static str = "unsigned int";

mod context;
mod variable;
mod declaration;
mod expression;
mod statement;
mod kernel_data;
mod kernel;

mod backend;
mod function_use_analyser;
mod parallel_for_variable_use;
