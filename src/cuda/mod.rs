pub mod writer;

pub trait CudaContext {
    fn generate_unique_identifier(&mut self) -> u32;
}

#[cfg(test)]
impl<F> CudaContext for F
    where F: FnMut() -> u32
{
    fn generate_unique_identifier(&mut self) -> u32 {
        self()
    }
}

pub mod declaration;
pub mod expression;
pub mod function;
pub mod kernel;

pub mod backend;
pub mod function_use_analyser;
pub mod parallel_for_variable_use;
