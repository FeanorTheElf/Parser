use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::super::language::position::NONEXISTING;
use super::writer::*;
use super::CudaContext;
use super::INDEX_TYPE;

// variable name for "normal" variables or functions: _<variable name>_<variable number>

/// array size variable name: d<dimension index>_<array name>_<array number>
const DIM_PREFIX: &'static str = "d";
/// kernel name: kernel_<kernel id>
const KERNEL_PREFIX: &'static str = "kernel";
/// used for assignment of function results to arrays, variable name: tmp_result resp. d<dimension index>_tmp_result
const TEMPORARY_ARRAY_RESULT: &'static str = "tmp_result";
/// used for starting kernels, parameter name: d<dimension index>_thread_offset
const THREAD_OFFSET_PREFIX: &'static str = "thread_offset";
/// used for passing how many threads used, parameter name: d<dimension index>_thread_acc_count
const THREAD_ACC_COUNT_PREFIX: &'static str = "thread_acc_count";

pub trait CudaWritableVariable {

    fn write_base(&self, out: &mut CodeWriter) -> Result<(), OutputError>;

    /// Writes the target language variable name that stores a part of the size information
    /// of this variable (only makes sense for arrays). The convention is that the size variable
    /// for dimension d contains the product of lengths in all dimensions d, d + 1, ..., n.
    /// Therefore, the variable for dimension 0 contains the total amount of elements in the array,
    /// and in a linear array memory representation, the entry at i0, ..., in can be found at
    /// i0 * dim_var_1 + i1 * dim_var_2 + ... + in
    fn write_dim(&self, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError>;

    fn write_kernel(id: u32, out: &mut CodeWriter) -> Result<(), OutputError>;

    /// Writes the target language name for the variable that contains the offset in the given
    /// dimension of the index variable (relativly to thread ids given by threadIdx). This is neceessary,
    /// as the index variables may be negative, the coordinates of the queued threads however cannot be
    fn write_thread_offset(kernel_id: u32, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError>;

    /// Writes the name for dim-th thread count variable for the kernel with id kernel_id; The contract
    /// is the same as in write_dim: dim 0 contains the total number of queued threads, dim 1 contains
    /// the number of threads for fixed first coordinate and so on.
    /// Formally, dim n contains the product thread_dimension(n) * ... * thread_dimension(dims)
    fn write_thread_acc_count(kernel_id: u32, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError>;
}

#[derive(Debug, PartialEq, Clone)]
pub enum CudaVariable<'a> {
    Base(&'a Name), TemporaryArrayResult(PrimitiveType, u32)
}

impl<'a> CudaVariable<'a> {
    pub fn calc_type<'stack, 'ast: 'stack>(&self, context: &dyn CudaContext<'stack, 'ast>) -> Type {
        match self {
            CudaVariable::Base(base) => context.calculate_var_type(base),
            CudaVariable::TemporaryArrayResult(base, dim) => Type::Array(*base, *dim)
        }
    }
}

impl<'a> CudaWritableVariable for CudaVariable<'a> {
    fn write_base(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            CudaVariable::Base(name) => if name.id != 0 {
                write!(out, "_{}_{}", name.name, name.id).map_err(OutputError::from)
            } else {
                write!(out, "_{}_", name.name).map_err(OutputError::from)
            },
            CudaVariable::TemporaryArrayResult(_, _) => write!(out, "{}", TEMPORARY_ARRAY_RESULT).map_err(OutputError::from)
        }
    }

    fn write_dim(&self, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            CudaVariable::Base(name) => if name.id != 0 {
                write!(out, "{}{}_{}_{}", DIM_PREFIX, dim, name.name, name.id).map_err(OutputError::from)
            } else {
                write!(out, "{}{}_{}_", DIM_PREFIX, dim, name.name).map_err(OutputError::from)
            },
            CudaVariable::TemporaryArrayResult(_, _) => write!(out, "{}{}_{}", DIM_PREFIX, dim, TEMPORARY_ARRAY_RESULT).map_err(OutputError::from)
        }
    }

    fn write_kernel(id: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "{}_{}", KERNEL_PREFIX, id).map_err(OutputError::from)
    }

    fn write_thread_offset(_kernel_id: u32, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "{}{}_{}", DIM_PREFIX, dim, THREAD_OFFSET_PREFIX).map_err(OutputError::from)
    }
    
    fn write_thread_acc_count(_kernel_id: u32, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "{}{}_{}", DIM_PREFIX, dim, THREAD_ACC_COUNT_PREFIX).map_err(OutputError::from)
    }
}

impl CudaWritableVariable for Name {
    fn write_base(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        CudaVariable::Base(self).write_base(out)
    }

    fn write_dim(&self, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
        CudaVariable::Base(self).write_dim(dim, out)
    }

    fn write_kernel(id: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
        CudaVariable::write_kernel(id, out)
    }

    fn write_thread_offset(kernel_id: u32, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
        CudaVariable::write_thread_offset(kernel_id, dim, out)
    }
    
    fn write_thread_acc_count(kernel_id: u32, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
        CudaVariable::write_thread_acc_count(kernel_id, dim, out)
    }
}
