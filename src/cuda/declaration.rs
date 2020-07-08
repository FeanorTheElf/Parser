use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::super::analysis::symbol::*;
use super::writer::*;

const DIM_PREFIX: &'static str = "d";
const KERNEL_PREFIX: &'static str = "kernel";
const THREAD_OFFSET_PREFIX: &'static str = "thread_offset";
const THREAD_ACC_COUNT_PREFIX: &'static str = "thread_acc_count";
const INDEX_TYPE: &'static str = "unsigned int";

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

impl CudaWritableVariable for Name {
    fn write_base(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        if self.id != 0 {
            write!(out, "_{}_{}", self.name, self.id).map_err(OutputError::from)
        } else {
            write!(out, "_{}_", self.name).map_err(OutputError::from)
        }
    }

    fn write_dim(&self, dim: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
        if self.id != 0 {
            write!(out, "{}{}_{}_{}", DIM_PREFIX, dim, self.name, self.id).map_err(OutputError::from)
        } else {
            write!(out, "{}{}_{}_", DIM_PREFIX, dim, self.name).map_err(OutputError::from)
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

pub trait CudaWritableDeclaration {
    fn write_as_param(&self, out: &mut CodeWriter) -> Result<(), OutputError>;
}

impl CudaWritableDeclaration for Declaration {
    fn write_as_param(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self.calc_type() {
            Type::TestType => panic!("TestType not valid"),
            Type::JumpLabel => CompileError::new(self.pos(), format!("Jump label not a valid parameter"), ErrorType::TypeError).throw(),
            Type::Primitive(PrimitiveType::Int) => {
                write!(out, "int ")?;
                self.variable.write_base(out)
            },
            Type::Function(_, _) => Err(OutputError::UnsupportedCode(self.pos().clone(), "Function parameters are not supported".to_owned())),
            Type::Array(_, _) => CompileError::new(self.pos(), format!("Arrays must be passed as views. Consider using &{}", self.variable), ErrorType::ArrayParameterPerValue).throw(),
            Type::View(viewn_type) => match &*viewn_type {
                Type::TestType => panic!("TestType not valid"),
                Type::JumpLabel => CompileError::new(self.pos(), format!("Jump label not a valid parameter"), ErrorType::TypeError).throw(),
                Type::Primitive(PrimitiveType::Int) => {
                    write!(out, "int* ")?;
                    self.variable.write_base(out)
                },
                Type::Function(_, _) => Err(OutputError::UnsupportedCode(self.pos().clone(), "Function parameters are not supported".to_owned())),
                Type::Array(PrimitiveType::Int, dim) => {
                    write!(out, "int* ")?;
                    self.variable.write_base(out)?;
                    for d in 0..*dim {
                        write!(out, ", ")?;
                        write!(out, "{}", INDEX_TYPE)?;
                        write!(out, " ")?;
                        self.variable.write_dim(d, out)?;
                    }
                    Ok(())
                },
                Type::View(_) => CompileError::new(self.pos(), format!("Nested views are not allowed"), ErrorType::ViewOnView).throw()
            }
        }
    }
}