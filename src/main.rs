#![feature(test)]
#![feature(trait_alias)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]

extern crate itertools;
extern crate take_mut;
#[cfg(test)]
extern crate test;

#[macro_use]

mod util;

#[macro_use]

mod language;

mod analysis;
mod cuda;
mod lexer;
mod parser;
mod transformation;

use language::backend::*;
use language::prelude::*;
use lexer::lexer::lex;
use parser::Parser;

fn main() {

    let mut program = Program::parse(&mut lex("

    fn set(a: &int, i: int,) {
        a = i;
    }
    
    fn main() {
        let a: int[,] = allocate1d(5,);
        pfor i: int, with this[i,], in a {
            set(a[i,], i,);
        }
        let b: int[,] = copy(a,);
        print(b,);
    }

    fn len(x: &int[,],): int native;

    fn allocate1d(len: int,): int[,] native;

    fn print(x: &int[,],) native;

    fn copy(x: &int[,],): int[,] {
        let result: int[,] = allocate1d(len(x,),);
        copy_array(result, x,);
        return result;
    }

    fn copy_array(dst: &int[,], src: &int[,],) {
        pfor i: int, with this[i,], in dst with this[i,], in src {
            dst[i,] = src[i,];
        }
    }
    
    "))
    .unwrap();

    let mut cuda_backend = cuda::backend::CudaBackend::new();

    cuda_backend.init().unwrap();

    cuda_backend.transform_program(&mut program).unwrap();

    let mut out: String = "".to_owned();

    let mut target: StringWriter = StringWriter::new(&mut out);

    let mut writer: CodeWriter = CodeWriter::new(&mut target);

    cuda_backend.generate(&program, &mut writer).unwrap();

    println!("{}", out);
}
