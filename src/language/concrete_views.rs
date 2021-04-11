use super::types::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewZeros {}

impl ViewFuncs for ViewZeros {}
impl View for ViewZeros {}

pub const VIEW_ZEROS: ViewZeros = ViewZeros {};