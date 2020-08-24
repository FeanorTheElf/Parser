use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::writer::*;

pub trait Writable {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError>;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CudaPrimitiveType {
    Int, Float, Bool, Void, VoidPtr, Index
}

impl Writable for CudaPrimitiveType {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            CudaPrimitiveType::Int => write!(out, "int").map_err(OutputError::from),
            CudaPrimitiveType::Float => write!(out, "float").map_err(OutputError::from),
            CudaPrimitiveType::Bool => write!(out, "bool").map_err(OutputError::from),
            CudaPrimitiveType::Void => write!(out, "void").map_err(OutputError::from),
            CudaPrimitiveType::VoidPtr => write!(out, "void").map_err(OutputError::from),
            CudaPrimitiveType::Index => write!(out, "unsigned int").map_err(OutputError::from)
        }
    }
}

#[derive(Clone)]
pub struct CudaType {
    pub constant: bool,
    pub base: CudaPrimitiveType,
    pub ptr_count: u32
}

impl Writable for CudaType {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        if self.constant {
            write!(out, "const ")?;
        }
        self.base.write(out)?;
        for _i in 0..self.ptr_count {
            write!(out, "*")?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub enum CudaIdentifier {
    ThreadIdxX, BlockIdxX, BlockDimX, GridDimX, SharedMem, TmpVar, TmpSizeVar(u32),
    /// name of variable containing the value of the source language variable with given identifier
    ValueVar(Name), 
    /// name of the variable containing the postfix product of sizes in the dimensions. Formally, this means
    /// that for an array of shape (s0, s1, ..., sn), the i-th array size var contains s(i+1) * ... * sn.
    ArraySizeVar(Name, u32), 
    /// name of the kernel with given id
    Kernel(u32),
    /// name of the variable containing the offset of the source thread grid to the cuda-defined thread
    /// grid always starting at 0. This is necessary, as in the source thread grid, one can e.g. execute threads for ids
    /// -20 to -10, but cuda ids always start at 0. Here, the first parameter is the kernel id and the second 
    /// parameter is the dimension
    ThreadGridOffsetVar(u32, u32),
    /// name of the variable containing the postfix product of the sizes of the source thread grid, so
    /// the contract is identical to the one of ArraySizeVar (see there for details). The first parameter
    /// is the kernel id and the second parameter the dimension.
    ThreadGridSizeVar(u32, u32)
}

impl Writable for CudaIdentifier {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            CudaIdentifier::ThreadIdxX => write!(out, "threadIdx.x").map_err(OutputError::from),
            CudaIdentifier::BlockIdxX => write!(out, "blockIdx.x").map_err(OutputError::from),
            CudaIdentifier::BlockDimX => write!(out, "blockDim.x").map_err(OutputError::from),
            CudaIdentifier::GridDimX => write!(out, "gridDim.x").map_err(OutputError::from),
            CudaIdentifier::SharedMem => write!(out, "sharedMem").map_err(OutputError::from),
            CudaIdentifier::ValueVar(name) => if name.id != 0 { 
                    write!(out, "{}_{}", name.name, name.id) .map_err(OutputError::from)
                } else { 
                    write!(out, "{}_", name.name).map_err(OutputError::from)
                },
            CudaIdentifier::ArraySizeVar(name, dim) => if name.id != 0 { 
                    write!(out, "{}_{}d{}", name.name, name.id, dim) .map_err(OutputError::from)
                } else { 
                    write!(out, "{}_d{}", name.name, dim).map_err(OutputError::from)
                },
            CudaIdentifier::Kernel(id) => write!(out, "kernel{}", id).map_err(OutputError::from),
            CudaIdentifier::ThreadGridOffsetVar(kernel_id, dim) => write!(out, "kernel{}o{}", kernel_id, dim).map_err(OutputError::from),
            CudaIdentifier::ThreadGridSizeVar(kernel_id, dim) => write!(out, "kernel{}d{}", kernel_id, dim).map_err(OutputError::from),
            CudaIdentifier::TmpVar => write!(out, "tmp").map_err(OutputError::from),
            CudaIdentifier::TmpSizeVar(dim) => write!(out, "tmpd{}", dim).map_err(OutputError::from)
        }
    }
}

pub trait CudaStatement : Writable {}

#[derive(Clone)]
pub enum AddSub {
    Plus, Minus
}

impl AddSub {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            AddSub::Plus => write!(out, " + ").map_err(OutputError::from),
            AddSub::Minus => write!(out, " - ").map_err(OutputError::from)
        }
    }

    fn write_unary(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            AddSub::Plus => Ok(()),
            AddSub::Minus => write!(out, "-").map_err(OutputError::from)
        }
    }
}

#[derive(Clone)]
pub enum MulDiv {
    Multiply, Divide
}

impl MulDiv {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            MulDiv::Multiply => write!(out, " * ").map_err(OutputError::from),
            MulDiv::Divide => write!(out, " / ").map_err(OutputError::from)
        }
    }

    fn write_unary(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            MulDiv::Multiply => Ok(()),
            MulDiv::Divide => write!(out, "1 / ").map_err(OutputError::from)
        }
    }
}

#[derive(Clone)]
pub enum Cmp {
    Eq, Neq, Ls, Gt, Leq, Geq
}

impl Cmp {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            Cmp::Eq => write!(out, " == ").map_err(OutputError::from),
            Cmp::Neq => write!(out, " != ").map_err(OutputError::from),
            Cmp::Ls => write!(out, " < ").map_err(OutputError::from),
            Cmp::Gt => write!(out, " > ").map_err(OutputError::from),
            Cmp::Leq => write!(out, " <= ").map_err(OutputError::from),
            Cmp::Geq => write!(out, " >= ").map_err(OutputError::from)
        }
    }
}

#[derive(Clone)]
pub enum CudaExpression {
    Call(CudaIdentifier, Vec<CudaExpression>),
    /// kernel_name, grid size, block size, shared mem size, parameters
    KernelCall(CudaIdentifier, Box<CudaExpression>, Box<CudaExpression>, Box<CudaExpression>, Vec<CudaExpression>),
    Sum(Vec<(AddSub, CudaExpression)>), 
    Identifier(CudaIdentifier), 
    Literal(i32),
    Product(Vec<(MulDiv, CudaExpression)>),
    Comparison(Cmp, Box<CudaExpression>, Box<CudaExpression>),
    Conjunction(Vec<CudaExpression>),
    Disjunction(Vec<CudaExpression>),
    Index(CudaIdentifier, Box<CudaExpression>),
    AddressOf(Box<CudaExpression>),
    Nullptr,
    // target index (for given dimension), 1d source index, strides, offsets
    MultiDimIndexCalculation(u32, Box<CudaExpression>, Vec<CudaExpression>, Vec<CudaExpression>)
}

impl CudaExpression {
    fn get_priority(&self) -> i32 {
        match self {
            CudaExpression::KernelCall(_, _, _, _, _) => i32::MIN + 1,
            CudaExpression::Conjunction(_) => -2,
            CudaExpression::Disjunction(_) => -2, 
            CudaExpression::Comparison(_, _, _) => -1,
            CudaExpression::MultiDimIndexCalculation(_, _, _, _) => 0,
            CudaExpression::Sum(_) => 0,
            CudaExpression::Product(_) => 1,
            CudaExpression::AddressOf(_) => 2,
            CudaExpression::Index(_, _) => 3,
            CudaExpression::Identifier(_) => i32::MAX,
            CudaExpression::Literal(_) => i32::MAX,
            CudaExpression::Nullptr => i32::MAX,
            CudaExpression::Call(_, _) => i32::MAX,
        }
    }

    fn write_expression(&self, parent_priority: i32, out: &mut CodeWriter) -> Result<(), OutputError> {
        let priority = self.get_priority();
        if priority <= parent_priority {
            write!(out, "(")?;
        }
        match self {
            CudaExpression::Call(func, params) => {
                func.write(out)?;
                write!(out, "(")?;
                out.write_comma_separated(params.iter().map(|p| move |out: &mut CodeWriter| p.write_expression(i32::MIN, out)))?;
                write!(out, ")")?;
            },
            CudaExpression::KernelCall(kernel, grid_size, block_size, shared_mem, params) => {
                kernel.write(out)?;
                write!(out, " <<< dim3(")?;
                grid_size.write_expression(priority, out)?;
                write!(out, "), dim3(")?;
                block_size.write_expression(priority, out)?;
                write!(out, "), ")?;
                shared_mem.write_expression(priority, out)?;
                write!(out, " >>> (")?;
                out.write_comma_separated(params.iter().map(|p| move |out: &mut CodeWriter| p.write_expression(priority, out)))?;
                write!(out, ")")?;
            },
            CudaExpression::Identifier(name) => {
                name.write(out)?;
            },
            CudaExpression::Literal(val) => {
                write!(out, "{}", val)?;
            },
            CudaExpression::Sum(summands) =>  
                out.write_many(summands.iter().enumerate().map(|(index, (operator, value))| move |out: &mut CodeWriter| if index == 0 {
                    operator.write_unary(out)?;
                    value.write_expression(priority, out)
                } else {
                    operator.write(out)?;
                    value.write_expression(priority, out)
                }))?,
            CudaExpression::Product(factors) =>  
                out.write_many(factors.iter().enumerate().map(|(index, (operator, value))| move |out: &mut CodeWriter| if index == 0 {
                    operator.write_unary(out)?;
                    value.write_expression(priority, out)
                } else {
                    operator.write(out)?;
                    value.write_expression(priority, out)
                }))?,
            CudaExpression::Comparison(operator, lhs, rhs) => {
                lhs.write_expression(priority, out)?;
                operator.write(out)?;
                rhs.write_expression(priority, out)?;
            },
            CudaExpression::Conjunction(values) => {
                out.write_separated(values.iter().map(|v| move |out: &mut CodeWriter| v.write_expression(priority, out)), |out: &mut CodeWriter| write!(out, " && ").map_err(OutputError::from))?
            },
            CudaExpression::Disjunction(values) => {
                out.write_separated(values.iter().map(|v| move |out: &mut CodeWriter| v.write_expression(priority, out)), |out: &mut CodeWriter| write!(out, " || ").map_err(OutputError::from))?
            }, 
            CudaExpression::Index(array, index) => {
                array.write(out)?;
                write!(out, "[")?;
                index.write_expression(i32::MIN, out)?;
                write!(out, "]")?;
            },
            CudaExpression::AddressOf(expr) => {
                write!(out, "&")?;
                expr.write_expression(priority, out)?;
            },
            CudaExpression::Nullptr => write!(out, "nullptr")?,
            CudaExpression::MultiDimIndexCalculation(dimension, index, strides, offsets)  => {
                // calculate the coordinates as queued thread from the one-dimensional index
                if *dimension > 0 {
                    write!(out, "(static_cast<int>(")?;
                    index.write(out)?;
                    write!(out, ") % ")?;
                    strides[*dimension as usize].write_expression(CudaExpression::Product(vec![]).get_priority(), out)?;
                    write!(out, ")")?;
                } else {
                    write!(out, "static_cast<int>(")?;
                    index.write(out)?;
                    write!(out, ")")?;
                }
                if *dimension as usize + 1 < strides.len() {
                    write!(out, " / ")?;
                    strides[*dimension as usize + 1].write_expression(CudaExpression::Product(vec![]).get_priority(), out)?;
                }
                // add the offset
                write!(out, " + ")?;
                offsets[*dimension as usize].write_expression(CudaExpression::Sum(vec![]).get_priority(), out)?;
            }
        };
        if priority <= parent_priority {
            write!(out, ")")?;
        }
        Ok(())
    }
}

impl Writable for CudaExpression {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        self.write_expression(i32::MIN, out)
    }
}
impl CudaStatement for CudaExpression {}

pub struct CudaBlock {
    pub statements: Vec<Box<dyn CudaStatement>>
}

impl Writable for CudaBlock {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        out.enter_block()?;
        out.write_separated(self.statements.iter().map(|s: &Box<dyn CudaStatement>| move |out: &mut CodeWriter| -> Result<(), OutputError> {
            s.write(out)?;
            write!(out, ";")?;
            Ok(())
        }), |out: &mut CodeWriter| out.newline().map_err(OutputError::from))?;
        out.exit_block()?;
        Ok(())
    }
}
impl CudaStatement for CudaBlock {}

pub struct CudaFunction {
    pub host: bool,
    pub device: bool,
    pub return_type: CudaType,
    pub params: Vec<(CudaType, CudaIdentifier)>,
    pub name: CudaIdentifier,
    pub body: CudaBlock
}

impl Writable for CudaFunction {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        if self.host {
            write!(out, "__host__ ")?;
        } 
        if self.device {
            write!(out, "__device__ ")?;
        }
        self.return_type.write(out)?;
        write!(out, " ")?;
        self.name.write(out)?;
        write!(out, "(")?;
        out.write_comma_separated(self.params.iter().map(|p: &(CudaType, CudaIdentifier)| move |out: &mut CodeWriter| -> Result<(), OutputError> {
            p.0.write(out)?;
            write!(out, " ")?;
            p.1.write(out)?;
            Ok(())
        }))?;
        write!(out, ")")?;
        self.body.write(out)?;
        Ok(())
    }
}

pub struct CudaKernel {
    pub params: Vec<(CudaType, CudaIdentifier)>,
    pub name: CudaIdentifier,
    pub body: CudaBlock
}

impl Writable for CudaKernel {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "__global__ void ")?;
        self.name.write(out)?;
        write!(out, "(")?;
        out.write_comma_separated(self.params.iter().map(|p: &(CudaType, CudaIdentifier)| move |out: &mut CodeWriter| -> Result<(), OutputError> {
            p.0.write(out)?;
            write!(out, " ")?;
            p.1.write(out)?;
            Ok(())
        }))?;
        write!(out, ") ")?;
        self.body.write(out)?;
        Ok(())
    }
}

pub struct CudaAssignment {
    pub assignee: CudaExpression,
    pub value: CudaExpression
}

impl Writable for CudaAssignment {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        self.assignee.write(out)?;
        write!(out, " = ")?;
        self.value.write(out)?;
        Ok(())
    }
}
impl CudaStatement for CudaAssignment {}

pub struct CudaLabel {
    pub name: CudaIdentifier
}

impl Writable for CudaLabel {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        self.name.write(out)?;
        write!(out, ": ")?;
        Ok(())
    }
}
impl CudaStatement for CudaLabel {}

pub struct CudaGoto {
    pub target: CudaIdentifier
}

impl Writable for CudaGoto {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "goto ")?;
        self.target.write(out)?;
        Ok(())
    }
}
impl CudaStatement for CudaGoto {}

pub struct CudaReturn {
    pub value: Option<CudaExpression>
}

impl Writable for CudaReturn {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "return ")?;
        if let Some(val) = &self.value {
            val.write(out)?;
        }
        Ok(())
    }
}
impl CudaStatement for CudaReturn {}

pub struct CudaIf {
    pub cond: CudaExpression,
    pub body: CudaBlock
}

impl Writable for CudaIf {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "if (")?;
        self.cond.write(out)?;
        write!(out, ") ")?;
        self.body.write(out)?;
        Ok(())
    }
}
impl CudaStatement for CudaIf {}

pub struct CudaWhile {
    pub cond: CudaExpression,
    pub body: CudaBlock
}

impl Writable for CudaWhile {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "while (")?;
        self.cond.write(out)?;
        write!(out, ") ")?;
        self.body.write(out)?;
        Ok(())
    }
}
impl CudaStatement for CudaWhile {}

pub struct CudaMemcpy {
    pub destination: CudaExpression, 
    pub source: CudaExpression, 
    pub length: CudaExpression,
    pub base_type: CudaType,
    pub device: bool
}

impl Writable for CudaMemcpy {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        if self.device {
            write!(out, "memcpy(")?;
        } else {
            write!(out, "checkCudaStatus(cudaMemcpy(")?;
        }
        self.destination.write(out)?;
        write!(out, ", ")?;
        self.source.write(out)?;
        write!(out, ", sizeof(")?;
        self.base_type.write(out)?;
        write!(out, ") * ")?;
        self.length.write_expression(CudaExpression::Product(vec![]).get_priority(), out)?;
        if self.device {
            write!(out, ")")?;
        } else {
            write!(out, ", cudaMemcpyDeviceToDevice))")?;
        }
        Ok(())
    }
}
impl CudaStatement for CudaMemcpy {}

pub struct CudaVarDeclaration {
    pub var: CudaIdentifier,
    pub value: Option<CudaExpression>,
    pub var_type: CudaType
}

impl Writable for CudaVarDeclaration {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        self.var_type.write(out)?;
        write!(out, " ")?;
        self.var.write(out)?;
        if let Some(val) = &self.value {
            write!(out, " = ")?;
            val.write(out)?;
        }
        Ok(())
    }
}
impl CudaStatement for CudaVarDeclaration {}

pub struct CudaAssert {
    pub expr: CudaExpression
}

impl Writable for CudaAssert {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "assert(")?;
        self.expr.write(out)?;
        write!(out, ")")?;
        Ok(())
    }
}
impl CudaStatement for CudaAssert {}
