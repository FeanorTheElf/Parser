use super::code_gen::*;
use super::super::language::prelude::*;

pub trait VariableStorageFuncs: std::any::Any + std::fmt::Debug {

    fn write_struct(&self, g: &mut dyn CodeGenerator) -> OutResult;
    fn write_init_from(&self, name: &str, rhs_name: &str, rhs: &dyn VariableStorage, g: &mut dyn BlockGenerator) -> OutResult;
    fn write_copy_from(&self, name: &str, rhs_name: &str, rhs: &dyn VariableStorage, g: &mut dyn BlockGenerator) -> OutResult;

    fn get_out_type(&self) -> OutType;
    fn get_type(&self) -> &Type;
    fn get_dims(&self) -> usize;
    fn get_aggregated_len(&self, name: &str, index: usize) -> OutExpression;
    fn get_entry_at(&self, name: &str, indices: Vec<OutExpression>) -> OutExpression;
}

dynamic_trait!{ VariableStorage: VariableStorageFuncs; VariableStorageDynCastable }