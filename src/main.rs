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

mod lexer;
mod parser;

mod analysis;
mod transformation;

mod cuda;

use language::backend::*;
use language::prelude::*;
use lexer::lexer::lex;
use parser::Parser;

fn main() {
    let mut program = Program::parse(&mut lex("
    
    fn main() {
        let a: int[,] = allocate1d(5,);
        let i: int = 0;
        while (i < 5) {
            a[i,] = i;
            i = i + 1;
        }
        let b: int[,] = copy(a,);
    }

    fn len(x: &int[,],): int native;

    fn allocate1d(len: int,): int[,] native;

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
