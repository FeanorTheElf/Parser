use super::super::language::prelude::*;

pub fn get_functions_to_export<'a>(program: &'a [Box<Function>]) -> impl Iterator<Item = &'a Function> {
    std::iter::once(&**program.iter().find(|func| func.identifier.name.as_str() == "main").unwrap())
}