#![feature(fn_traits)]
#![feature(specialization)]
#![feature(never_type)]
#![feature(test)]
#![allow(unused)]

#[cfg(test)]
extern crate test;

#[macro_use]
mod util;
mod lexer;
mod parser;
#[macro_use]
mod transformer;
mod language;
mod la;
mod backend;

use lexer::tokens::*;
use parser::Parse;
use lexer::lexer::lex;
use parser::prelude::*;
use language::scope::{ ScopeTable, annotate_sope_info_func };
use language::symbol::{ SymbolTable, annotate_symbols_function };
use transformer::*;

struct TestTransformer();

impl<'a> SpecificLifetimeTransformer<'a, ()> for TestTransformer {
	type Prepared = TestTransformer;

    fn prepare(self, program: &'a Program, data: ()) -> Result<Self::Prepared, CompileError> {
		Ok(self)
	}
}

impl PreparedTransformer for TestTransformer {
	fn transform(self, program: &Program) -> Result<(), CompileError> {
		Ok(())
	}
}

impl ChainablePreparedTransformer<TestTransformer> for TestTransformer {
	fn transform_chained(self, program: &Program, next: TestTransformer) -> Result<(), CompileError> {
		next.transform(program)
	}
}

fn main() {
	let test_transformer = TestTransformer();
	let program: Program = unimplemented!();
	let foo: Result<TestTransformer, CompileError> = test_transformer.prepare(&program, ());
	let bar = foo.unwrap();
	bar.transform_chained(&program, TestTransformer());
}
