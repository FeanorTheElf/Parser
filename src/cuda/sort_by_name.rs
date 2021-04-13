use super::super::language::prelude::*;

pub struct SortByName;

impl<'a, 'b, 'c> FnOnce<(&'b &'a dyn SymbolDefinition, &'c &'a dyn SymbolDefinition)>
    for SortByName
{
    type Output = std::cmp::Ordering;

    extern "rust-call" fn call_once(
        self,
        values: (&'b &'a dyn SymbolDefinition, &'c &'a dyn SymbolDefinition),
    ) -> std::cmp::Ordering {
        self.call(values)
    }
}

impl<'a, 'b, 'c> FnMut<(&'b &'a dyn SymbolDefinition, &'c &'a dyn SymbolDefinition)>
    for SortByName
{
    extern "rust-call" fn call_mut(
        &mut self,
        values: (&'b &'a dyn SymbolDefinition, &'c &'a dyn SymbolDefinition),
    ) -> std::cmp::Ordering {
        self.call(values)
    }
}

impl<'a, 'b, 'c> Fn<(&'b &'a dyn SymbolDefinition, &'c &'a dyn SymbolDefinition)> for SortByName {

    extern "rust-call" fn call(
        &self,
        (lhs, rhs): (&'b &'a dyn SymbolDefinition, &'c &'a dyn SymbolDefinition),
    ) -> std::cmp::Ordering {
        lhs.get_name().name.cmp(&rhs.get_name().name)
    }
}

impl std::default::Default for SortByName {

    fn default() -> Self {
        SortByName
    }
}