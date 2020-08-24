use super::language::prelude::*;
use super::analysis::scope::*;
use super::analysis::symbol::*;
use super::analysis::types::*;

pub mod writer;

pub trait CudaContext<'stack, 'ast: 'stack> {
    fn generate_unique_identifier(&mut self) -> u32;
    fn is_device_context(&self) -> bool;
    fn set_device(&mut self);
    fn set_host(&mut self);
    fn set_scope(&mut self, scopes: DefinitionScopeStack<'stack, 'ast>);
    fn get_scopes(&self) -> &DefinitionScopeStack<'stack, 'ast>;
    fn get_current_function(&self) -> &Function;
    fn set_current_function(&mut self, function: &'ast Function);

    fn calculate_type(&self, expr: &Expression) -> Type {
        expr.calculate_type(self.get_scopes())
    }

    fn calculate_var_type(&self, variable: &Name) -> Type {
        self.get_scopes().get(variable).expect(format!("Expected {} to be defined", variable).as_str()).calc_type()
    }
}

impl<'stack, 'ast: 'stack, T: CudaContext<'stack, 'ast>> CudaContext<'stack, 'ast> for Box<T> {
    fn generate_unique_identifier(&mut self) -> u32 {
        (**self).generate_unique_identifier()
    }

    fn is_device_context(&self) -> bool {
        (**self).is_device_context()
    }

    fn set_device(&mut self) {
        (**self).set_device()
    }

    fn set_host(&mut self) {
        (**self).set_host()
    }
    
    fn set_scope(&mut self, scopes: DefinitionScopeStack<'stack, 'ast>) {
        (**self).set_scope(scopes)
    }

    fn get_scopes(&self) -> &DefinitionScopeStack<'stack, 'ast> {
        (**self).get_scopes()
    }

    fn calculate_var_type(&self, variable: &Name) -> Type {
        (**self).calculate_var_type(variable)
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
}

const INDEX_TYPE: &'static str = "unsigned int";

mod context;
mod ast;
mod statement;
mod kernel_data;
mod kernel;
