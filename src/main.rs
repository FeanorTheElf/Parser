#![feature(test)]
#![feature(trait_alias)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(iter_map_while)]
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

// mod analysis;
// mod lexer;
// mod parser;
// mod transformation;

// #[macro_use]
// mod cuda;

// mod cli;

fn main() {
}
