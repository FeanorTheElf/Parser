use super::super::language::backend::*;
use super::super::language::prelude::*;
use feanor_la::rat::{r64, ZERO, ONE};
use std::ops::{Add, Mul, Sub};

pub trait Writable {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError>;
}

#[derive(Clone, Copy, PartialEq, Eq)]

pub enum CudaPrimitiveType {
    Int,
    Float,
    Bool,
    Void,
    VoidPtr,
    Index,
}

impl Writable for CudaPrimitiveType {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        match self {
            CudaPrimitiveType::Int => write!(out, "int").map_err(OutputError::from),
            CudaPrimitiveType::Float => write!(out, "float").map_err(OutputError::from),
            CudaPrimitiveType::Bool => write!(out, "bool").map_err(OutputError::from),
            CudaPrimitiveType::Void => write!(out, "void").map_err(OutputError::from),
            CudaPrimitiveType::VoidPtr => write!(out, "void").map_err(OutputError::from),
            CudaPrimitiveType::Index => write!(out, "unsigned int").map_err(OutputError::from),
        }
    }
}

#[derive(Clone)]

pub struct CudaType {
    pub constant: bool,
    pub base: CudaPrimitiveType,
    pub owned: bool,
    pub ptr: bool
}

impl Writable for CudaType {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        if self.constant {
            write!(out, "const ")?;
        }

        if self.owned {
            write!(out, "DevPtr<")?;
            self.base.write(out)?;
            write!(out, ">")?;
        } else {
            self.base.write(out)?;
        }

        if self.ptr {
            write!(out, "*")?;
        }

        Ok(())
    }
}

#[derive(Clone)]

pub enum CudaIdentifier {
    ThreadIdxX,
    BlockIdxX,
    BlockDimX,
    GridDimX,
    SharedMem,
    TmpVar,
    TmpSizeVar(u32),
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
    ThreadGridSizeVar(u32, u32),
    /// in some situations, we do not need the postfix product of the array sizes, but the array sizes themselves.
    /// In this case, use this identifier, but usually, it must be declared and initialized correctly before.
    /// have local_array_id, dimension
    TmpArrayShapeVar(u32, u32),
    OutputValueVar,
    OutputArraySizeVar(u32),
}

impl Writable for CudaIdentifier {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        match self {
            CudaIdentifier::ThreadIdxX => write!(out, "threadIdx.x").map_err(OutputError::from),
            CudaIdentifier::BlockIdxX => write!(out, "blockIdx.x").map_err(OutputError::from),
            CudaIdentifier::BlockDimX => write!(out, "blockDim.x").map_err(OutputError::from),
            CudaIdentifier::GridDimX => write!(out, "gridDim.x").map_err(OutputError::from),
            CudaIdentifier::SharedMem => write!(out, "sharedMem").map_err(OutputError::from),
            CudaIdentifier::ValueVar(name) => {
                if name.id != 0 {

                    write!(out, "{}_{}", name.name, name.id).map_err(OutputError::from)
                } else {

                    write!(out, "{}_", name.name).map_err(OutputError::from)
                }
            }
            CudaIdentifier::ArraySizeVar(name, dim) => {
                if name.id != 0 {

                    write!(out, "{}_{}d{}", name.name, name.id, dim).map_err(OutputError::from)
                } else {

                    write!(out, "{}_d{}", name.name, dim).map_err(OutputError::from)
                }
            }
            CudaIdentifier::Kernel(id) => write!(out, "kernel{}", id).map_err(OutputError::from),
            CudaIdentifier::ThreadGridOffsetVar(kernel_id, dim) => {
                write!(out, "kernel{}o{}", kernel_id, dim).map_err(OutputError::from)
            }
            CudaIdentifier::ThreadGridSizeVar(kernel_id, dim) => {
                write!(out, "kernel{}d{}", kernel_id, dim).map_err(OutputError::from)
            }
            CudaIdentifier::TmpVar => write!(out, "tmp").map_err(OutputError::from),
            CudaIdentifier::TmpSizeVar(dim) => {
                write!(out, "tmpd{}", dim).map_err(OutputError::from)
            }
            CudaIdentifier::TmpArrayShapeVar(array_id, dim) => {
                write!(out, "array{}shape{}", array_id, dim).map_err(OutputError::from)
            }
            CudaIdentifier::OutputValueVar => write!(out, "result").map_err(OutputError::from),
            CudaIdentifier::OutputArraySizeVar(dim) => {
                write!(out, "result{}", dim).map_err(OutputError::from)
            }
        }
    }
}

pub trait CudaStatement: Writable {}

#[derive(Clone, PartialEq, Eq)]

pub enum AddSub {
    Plus,
    Minus,
}

impl AddSub {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        match self {
            AddSub::Plus => write!(out, " + ").map_err(OutputError::from),
            AddSub::Minus => write!(out, " - ").map_err(OutputError::from),
        }
    }

    fn write_unary(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        match self {
            AddSub::Plus => Ok(()),
            AddSub::Minus => write!(out, "-").map_err(OutputError::from),
        }
    }

    fn toggle(&self) -> AddSub {

        match self {
            AddSub::Plus => AddSub::Minus,
            AddSub::Minus => AddSub::Plus,
        }
    }
}

#[derive(Clone, PartialEq, Eq)]

pub enum MulDiv {
    Multiply,
    Divide,
}

impl MulDiv {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        match self {
            MulDiv::Multiply => write!(out, " * ").map_err(OutputError::from),
            MulDiv::Divide => write!(out, " / ").map_err(OutputError::from),
        }
    }

    fn write_unary(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        match self {
            MulDiv::Multiply => Ok(()),
            MulDiv::Divide => write!(out, "1 / ").map_err(OutputError::from),
        }
    }
}

#[derive(Clone)]

pub enum Cmp {
    Eq,
    Neq,
    Ls,
    Gt,
    Leq,
    Geq,
}

impl Cmp {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        match self {
            Cmp::Eq => write!(out, " == ").map_err(OutputError::from),
            Cmp::Neq => write!(out, " != ").map_err(OutputError::from),
            Cmp::Ls => write!(out, " < ").map_err(OutputError::from),
            Cmp::Gt => write!(out, " > ").map_err(OutputError::from),
            Cmp::Leq => write!(out, " <= ").map_err(OutputError::from),
            Cmp::Geq => write!(out, " >= ").map_err(OutputError::from),
        }
    }
}

#[derive(Clone)]

pub enum CudaExpression {
    Call(CudaIdentifier, Vec<CudaExpression>),
    Sum(Vec<(AddSub, CudaExpression)>),
    Identifier(CudaIdentifier),
    IntLiteral(i64),
    FloatLiteral(f64),
    RatLiteral(r64),
    Product(Vec<(MulDiv, CudaExpression)>),
    Comparison(Cmp, Box<CudaExpression>, Box<CudaExpression>),
    Conjunction(Vec<CudaExpression>),
    Disjunction(Vec<CudaExpression>),
    Index(CudaIdentifier, Box<CudaExpression>),
    AddressOf(Box<CudaExpression>),
    Deref(Box<CudaExpression>),
    Nullptr,
    // target index (for given dimension), 1d source index, strides, offsets
    MultiDimIndexCalculation(
        u32,
        Box<CudaExpression>,
        Vec<CudaExpression>,
        Vec<CudaExpression>,
    ),
    Min(Vec<CudaExpression>),
    Round(Box<CudaExpression>),
    Max(Vec<CudaExpression>),
    IndexFloorDiv(Box<CudaExpression>, Box<CudaExpression>),
    Move(Box<CudaExpression>)
}

impl CudaExpression {
    pub fn is_constant_zero(&self) -> bool {

        match &self {
            CudaExpression::IntLiteral(x) => *x == 0,
            CudaExpression::FloatLiteral(x) => *x == 0.,
            CudaExpression::RatLiteral(val) => *val == ZERO,
            CudaExpression::Sum(summands) => summands.len() == 0,
            _ => false,
        }
    }

    pub fn is_constant_one(&self) -> bool {

        match &self {
            CudaExpression::IntLiteral(x) => *x == 1,
            CudaExpression::FloatLiteral(x) => *x == 1.,
            CudaExpression::RatLiteral(val) => *val == ONE,
            CudaExpression::Product(factors) => factors.len() == 0,
            _ => false,
        }
    }

    pub fn deref(target: CudaExpression) -> CudaExpression {

        match target {
            CudaExpression::AddressOf(result) => *result,
            target => CudaExpression::Deref(Box::new(target)),
        }
    }
}

impl Add for CudaExpression {
    type Output = CudaExpression;

    fn add(mut self, mut rhs: CudaExpression) -> CudaExpression {

        if self.is_constant_zero() {

            return rhs;
        } else if rhs.is_constant_zero() {

            return self;
        }

        match &mut self {
            CudaExpression::Sum(summands) => {

                summands.push((AddSub::Plus, rhs));

                self
            }
            _ => match &mut rhs {
                CudaExpression::Sum(summands) => {

                    summands.push((AddSub::Plus, self));

                    rhs
                }
                _ => CudaExpression::Sum(vec![(AddSub::Plus, self), (AddSub::Plus, rhs)]),
            },
        }
    }
}

impl Sub for CudaExpression {
    type Output = CudaExpression;

    fn sub(mut self, mut rhs: CudaExpression) -> CudaExpression {

        if rhs.is_constant_zero() {

            return self;
        }

        match &mut self {
            CudaExpression::Sum(summands) => {

                summands.push((AddSub::Minus, rhs));

                self
            }
            _ => match &mut rhs {
                CudaExpression::Sum(summands) => {

                    for (pm, _) in summands.iter_mut() {

                        *pm = pm.toggle();
                    }

                    summands.insert(0, (AddSub::Plus, self));

                    rhs
                }
                _ => CudaExpression::Sum(vec![(AddSub::Plus, self), (AddSub::Minus, rhs)]),
            },
        }
    }
}

impl Mul for CudaExpression {
    type Output = CudaExpression;

    fn mul(mut self, mut rhs: CudaExpression) -> CudaExpression {

        if self.is_constant_zero() {

            return CudaExpression::IntLiteral(0);
        } else if rhs.is_constant_zero() {

            return CudaExpression::IntLiteral(0);
        } else if self.is_constant_one() {

            return rhs;
        } else if rhs.is_constant_one() {

            return self;
        }

        match &mut self {
            CudaExpression::Product(summands) => {

                summands.push((MulDiv::Multiply, rhs));

                self
            }
            _ => match &mut rhs {
                CudaExpression::Product(summands) => {

                    summands.push((MulDiv::Multiply, self));

                    rhs
                }
                _ => {
                    CudaExpression::Product(vec![(MulDiv::Multiply, self), (MulDiv::Multiply, rhs)])
                }
            },
        }
    }
}

impl From<r64> for CudaExpression {
    fn from(rhs: r64) -> CudaExpression {

        CudaExpression::RatLiteral(rhs)
    }
}

impl<'a> From<&'a r64> for CudaExpression {
    fn from(rhs: &'a r64) -> CudaExpression {

        CudaExpression::RatLiteral(*rhs)
    }
}

impl CudaExpression {
    fn get_priority(&self) -> i32 {

        match self {
            CudaExpression::Conjunction(_) => -2,
            CudaExpression::Disjunction(_) => -2,
            CudaExpression::Comparison(_, _, _) => -1,
            CudaExpression::MultiDimIndexCalculation(_, _, _, _) => 0,
            CudaExpression::Sum(_) => 0,
            CudaExpression::IndexFloorDiv(_, _) => 0,
            CudaExpression::Product(_) => 1,
            CudaExpression::RatLiteral(_) => 1,
            CudaExpression::AddressOf(_) => 2,
            CudaExpression::Deref(_) => 2,
            CudaExpression::Index(_, _) => 3,
            CudaExpression::Identifier(_) => i32::MAX,
            CudaExpression::IntLiteral(_) => i32::MAX,
            CudaExpression::FloatLiteral(_) => i32::MAX,
            CudaExpression::Nullptr => i32::MAX,
            CudaExpression::Call(_, _) => i32::MAX,
            CudaExpression::Min(_) => i32::MAX,
            CudaExpression::Max(_) => i32::MAX,
            CudaExpression::Round(_) => i32::MAX,
            CudaExpression::Move(_) => i32::MAX,
        }
    }

    fn write_expression(
        &self,
        parent_priority: i32,
        out: &mut CodeWriter,
    ) -> Result<(), OutputError> {

        let priority = self.get_priority();

        if priority <= parent_priority {

            write!(out, "(")?;
        }

        match self {
            CudaExpression::Call(func, params) => {

                func.write(out)?;

                write!(out, "(")?;

                out.write_comma_separated(
                    params
                        .iter()
                        .map(|p| move |out: &mut CodeWriter| p.write_expression(i32::MIN, out)),
                )?;

                write!(out, ")")?;
            }
            CudaExpression::Identifier(name) => {

                name.write(out)?;
            }
            CudaExpression::IntLiteral(val) => {

                write!(out, "{}", val)?;
            }
            CudaExpression::FloatLiteral(val) => {

                write!(out, "{}.", val)?;
            }
            CudaExpression::RatLiteral(mut rhs) => {
                rhs.reduce();
                write!(out, "{}./{}.", rhs.num(), rhs.den())?;
            }
            CudaExpression::Sum(summands) => {
                if summands.len() == 0 {

                    write!(out, "0")?;
                } else {

                    out.write_many(summands.iter().enumerate().map(
                        |(index, (operator, value))| {

                            move |out: &mut CodeWriter| {
                                if index == 0 {

                                    operator.write_unary(out)?;

                                    value.write_expression(priority, out)
                                } else {

                                    operator.write(out)?;

                                    value.write_expression(priority, out)
                                }
                            }
                        },
                    ))?;
                }
            }
            CudaExpression::Product(factors) => {
                if factors.len() == 0 {

                    write!(out, "1")?;
                } else {

                    out.write_many(factors.iter().enumerate().map(
                        |(index, (operator, value))| {

                            move |out: &mut CodeWriter| {
                                if index == 0 {

                                    operator.write_unary(out)?;

                                    value.write_expression(priority, out)
                                } else {

                                    operator.write(out)?;

                                    value.write_expression(priority, out)
                                }
                            }
                        },
                    ))?;
                }
            }
            CudaExpression::Comparison(operator, lhs, rhs) => {

                lhs.write_expression(priority, out)?;

                operator.write(out)?;

                rhs.write_expression(priority, out)?;
            }
            CudaExpression::Conjunction(values) => out.write_separated(
                values
                    .iter()
                    .map(|v| move |out: &mut CodeWriter| v.write_expression(priority, out)),
                |out: &mut CodeWriter| write!(out, " && ").map_err(OutputError::from),
            )?,
            CudaExpression::Disjunction(values) => out.write_separated(
                values
                    .iter()
                    .map(|v| move |out: &mut CodeWriter| v.write_expression(priority, out)),
                |out: &mut CodeWriter| write!(out, " || ").map_err(OutputError::from),
            )?,
            CudaExpression::Index(array, index) => {

                array.write(out)?;

                write!(out, "[")?;

                index.write_expression(i32::MIN, out)?;

                write!(out, "]")?;
            }
            CudaExpression::AddressOf(expr) => {

                write!(out, "&")?;

                expr.write_expression(priority, out)?;
            }
            CudaExpression::Deref(expr) => {

                write!(out, "*")?;

                expr.write_expression(priority, out)?;
            }
            CudaExpression::Nullptr => write!(out, "nullptr")?,
            CudaExpression::MultiDimIndexCalculation(dimension, index, strides, offsets) => {

                // calculate the coordinates as queued thread from the one-dimensional index
                if *dimension > 0 {

                    write!(out, "(static_cast<int>(")?;

                    index.write(out)?;

                    write!(out, ") % ")?;

                    strides[*dimension as usize]
                        .write_expression(CudaExpression::Product(vec![]).get_priority(), out)?;

                    write!(out, ")")?;
                } else {

                    write!(out, "static_cast<int>(")?;

                    index.write(out)?;

                    write!(out, ")")?;
                }

                if *dimension as usize + 1 < strides.len() {

                    write!(out, " / ")?;

                    strides[*dimension as usize + 1]
                        .write_expression(CudaExpression::Product(vec![]).get_priority(), out)?;
                }

                // add the offset
                write!(out, " + ")?;

                offsets[*dimension as usize]
                    .write_expression(CudaExpression::Sum(vec![]).get_priority(), out)?;
            }
            CudaExpression::Min(exprs) => {

                write!(out, "min(")?;

                out.write_comma_separated(
                    exprs.iter().map(|expr| {

                        move |out: &mut CodeWriter| expr.write_expression(i32::MIN, out)
                    }),
                )?;

                write!(out, ")")?;
            }
            CudaExpression::Max(exprs) => {

                write!(out, "max(")?;

                out.write_comma_separated(
                    exprs.iter().map(|expr| {

                        move |out: &mut CodeWriter| expr.write_expression(i32::MIN, out)
                    }),
                )?;

                write!(out, ")")?;
            }
            CudaExpression::Round(expr) => {

                write!(out, "round(")?;

                expr.write_expression(i32::MIN, out)?;

                write!(out, ")")?;
            }
            CudaExpression::Move(expr) => {

                write!(out, "std::move(")?;

                expr.write_expression(i32::MIN, out)?;

                write!(out, ")")?;
            }
            CudaExpression::IndexFloorDiv(divident, divisor) => {

                write!(out, "(")?;

                divident.write_expression(priority, out)?;

                write!(out, " - 1) / ")?;

                divisor.write_expression(CudaExpression::Product(vec![]).get_priority(), out)?;

                write!(out, " + 1")?;
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
    pub statements: Vec<Box<dyn CudaStatement>>,
}

impl Writable for CudaBlock {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        out.enter_block()?;

        out.write_separated(
            self.statements.iter().map(|s: &Box<dyn CudaStatement>| {

                move |out: &mut CodeWriter| -> Result<(), OutputError> {

                    s.write(out)?;

                    write!(out, ";")?;

                    Ok(())
                }
            }),
            |out: &mut CodeWriter| out.newline().map_err(OutputError::from),
        )?;

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
    pub body: CudaBlock,
}

impl Writable for CudaFunction {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        out.newline()?;

        out.newline()?;

        if self.host {

            write!(out, "__host__ ")?;
        }

        if self.device {

            write!(out, "__device__ ")?;
        }

        write!(out, "inline ")?;

        self.return_type.write(out)?;

        write!(out, " ")?;

        self.name.write(out)?;

        write!(out, "(")?;

        out.write_comma_separated(self.params.iter().map(|p: &(CudaType, CudaIdentifier)| {

            move |out: &mut CodeWriter| -> Result<(), OutputError> {

                p.0.write(out)?;

                write!(out, " ")?;

                p.1.write(out)?;

                Ok(())
            }
        }))?;

        write!(out, ") ")?;

        self.body.write(out)?;

        Ok(())
    }
}

pub struct CudaKernel {
    pub params: Vec<(CudaType, CudaIdentifier)>,
    pub name: CudaIdentifier,
    pub body: CudaBlock,
}

impl Writable for CudaKernel {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        out.newline()?;

        out.newline()?;

        write!(out, "__global__ inline void ")?;

        self.name.write(out)?;

        write!(out, "(")?;

        out.write_comma_separated(self.params.iter().map(|p: &(CudaType, CudaIdentifier)| {

            move |out: &mut CodeWriter| -> Result<(), OutputError> {

                p.0.write(out)?;

                write!(out, " ")?;

                p.1.write(out)?;

                Ok(())
            }
        }))?;

        write!(out, ") ")?;

        self.body.write(out)?;

        Ok(())
    }
}

pub struct CudaAssignment {
    pub assignee: CudaExpression,
    pub value: CudaExpression,
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

pub struct CudaReturn {
    pub value: Option<CudaExpression>,
}

impl Writable for CudaReturn {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        write!(out, "return")?;

        if let Some(val) = &self.value {

            write!(out, " ")?;

            val.write(out)?;
        }

        Ok(())
    }
}

impl CudaStatement for CudaReturn {}

pub struct CudaLabel {
    pub name: CudaIdentifier,
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
    pub target: CudaIdentifier,
}

impl Writable for CudaGoto {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        write!(out, "goto ")?;

        self.target.write(out)?;

        Ok(())
    }
}

impl CudaStatement for CudaGoto {}

pub struct CudaIf {
    pub cond: CudaExpression,
    pub body: CudaBlock,
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
    pub body: CudaBlock,
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
    pub base_type: CudaPrimitiveType,
    pub device: bool,
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

        self.length
            .write_expression(CudaExpression::Product(vec![]).get_priority(), out)?;

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
    pub var_type: CudaType,
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
    pub expr: CudaExpression,
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

pub struct CudaKernelCall {
    pub name: CudaIdentifier,
    pub grid_size: CudaExpression,
    pub block_size: CudaExpression,
    pub shared_mem_size: CudaExpression,
    pub params: Vec<CudaExpression>,
}

impl Writable for CudaKernelCall {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        self.name.write(out)?;

        write!(out, " <<< dim3(")?;

        self.grid_size.write_expression(i32::MIN, out)?;

        write!(out, "), dim3(")?;

        self.block_size.write_expression(i32::MIN, out)?;

        write!(out, "), ")?;

        self.shared_mem_size.write_expression(i32::MIN, out)?;

        write!(out, " >>> (")?;

        out.write_comma_separated(
            self.params
                .iter()
                .map(|p| move |out: &mut CodeWriter| p.write(out)),
        )?;

        write!(out, ")")?;

        Ok(())
    }
}

impl CudaStatement for CudaKernelCall {}

pub struct CudaAlloc {
    pub device: bool,
    pub elements: CudaExpression,
    pub base_type: CudaType,
    pub ptr: CudaIdentifier,
}

impl Writable for CudaAlloc {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        if self.device {
            self.ptr.write(out)?;
            write!(out, " = malloc(sizeof(")?;
            
            self.base_type.write(out)?;
    
            write!(out, ") * ")?;
    
            self.elements
                .write_expression(CudaExpression::Product(vec![]).get_priority(), out)?;
            write!(out, ")")?;
        } else {
            write!(out, "checkCudaStatus(cudaMalloc(&")?;

            self.ptr.write(out)?;
    
            write!(out, ", sizeof(")?;
    
            self.base_type.write(out)?;
    
            write!(out, ") * ")?;
    
            self.elements
                .write_expression(CudaExpression::Product(vec![]).get_priority(), out)?;
    
            write!(out, "))")?;
        }

        Ok(())
    }
}

impl CudaStatement for CudaAlloc {}

pub struct CudaFree {
    pub device: bool,
    pub ptr: CudaIdentifier,
}

impl Writable for CudaFree {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {

        if self.device {
           write!(out, "free(")?;
           self.ptr.write(out)?;
           write!(out, ")")?;
        } else {
            write!(out, "checkCudaStatus(cudaFree(")?;
            self.ptr.write(out)?;
            write!(out, "))")?;
        }

        Ok(())
    }
}
impl CudaStatement for CudaFree {}