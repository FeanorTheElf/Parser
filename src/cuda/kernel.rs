use super::super::language::prelude::*;
use super::super::analysis::symbol::*;
use super::super::util::ref_eq::*;

use std::collections::HashSet;

struct KernelInfo<'a> {
    pfor: &'a ParallelFor,
    used_variables: HashSet<Ref<'a, dyn 'a + SymbolDefinition>>
}