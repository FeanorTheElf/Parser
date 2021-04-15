use super::super::language::prelude::*;
use super::gwh_str::*;
use super::code_gen::*;
use super::repr::*;

pub fn convert_base_type(primitive: PrimitiveType) -> OutPrimitiveType {
    match primitive {
        PrimitiveType::Int => OutPrimitiveType::Int,
        PrimitiveType::Bool => OutPrimitiveType::Bool,
        PrimitiveType::Float => OutPrimitiveType::Double
    }
}

pub fn aggregate_lengths(lengths: &Vec<OutExpression>, index: usize) -> OutExpression {
    assert!(index < lengths.len());
    OutExpression::prod(lengths.iter().take(index + 1).cloned())
}