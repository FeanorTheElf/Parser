#![feature(test)]
#![feature(trait_alias)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(iter_map_while)]
#![feature(min_const_generics)]
#![feature(concat_idents)]
#![feature(never_type)]

extern crate take_mut;

#[macro_use] 
extern crate serde;
extern crate serde_json;

#[cfg(test)]
extern crate test;

#[macro_use]
mod util;

#[macro_use]
mod language;

mod analysis;
mod lexer;
mod parser;
mod transformation;

#[macro_use]
mod cuda;

mod cli;

use cli::cli_backend::*;
use cli::cuda::*;

use language::prelude::*;
use parser::Parser;
use lexer::lexer::fragment_lex;

fn main() {
    let a = Statement::parse(&mut fragment_lex("let a: int = b(c, 0,);"), &mut TypeVec::new()).unwrap();
    assert_eq!(a.names().map(|n| n.name.as_str()).collect::<Vec<_>>(), vec!["a", "c", "b"]);
    create_cuda_exe_backend().run(MultiStageBackendOptions::from("E:\\Users\\Simon\\Documents\\Projekte\\Parser\\example")).unwrap();
}
