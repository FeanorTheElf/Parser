
pub trait OutError: std::any::Any + std::fmt::Debug {}

impl OutError for std::io::Error {}

impl From<std::io::Error> for Box<dyn OutError> {

    fn from(err: std::io::Error) -> Box<dyn OutError> {
        Box::new(err)
    }
}

pub type OutResult = Result<(), Box<dyn OutError>>;

pub trait CodeGenerator {

    fn write_function<'a, 'b>(
        &'a mut self, 
        params: Vec<(OutType, String)>, 
        return_type: Option<OutType>, 
        body: Box<dyn 'b + FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>,
        device_called: bool,
        host_called: bool
    ) -> OutResult;

    fn write_struct(
        &mut self, 
        name: String,
        vars: Vec<(OutType, String)>
    ) -> OutResult;
}

pub trait BlockGenerator {

    fn write_range_assign(&mut self, target_ty: OutType, target: OutExpression, source_ty: OutType, source: OutExpression, len: OutExpression) -> OutResult;
    fn write_variable_declaration(&mut self, name: String, ty: OutType, value: Option<OutExpression>) -> OutResult;
    fn write_return(&mut self, value: Option<OutExpression>) -> OutResult;
    fn write_assert(&mut self, value: OutExpression) -> OutResult;
    fn write_expr_statement(&mut self, expr: OutExpression) -> OutResult;
    fn write_entry_assign(&mut self, ty: OutType, arr: OutExpression, index: OutExpression, val: OutExpression) -> OutResult;
    fn write_value_assign(&mut self, ty: OutType, assignee: OutExpression, val: OutExpression) -> OutResult;

    fn write_if<'b>(
        &mut self, 
        condition: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult;
    
    fn write_block<'b>(
        &mut self, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult;

    fn write_while<'b>(
        &mut self, 
        condition: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult;

    fn write_integer_for<'b>(
        &mut self, 
        name: String,
        init: OutExpression,
        limit: OutExpression,
        increment: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult;

    fn write_parallel_code<'b>(
        &mut self, 
        thread_count: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>, OutExpression) -> OutResult>,
        used_outer_vars: Vec<(OutType, String)>
    ) -> OutResult;
}

impl<'c> BlockGenerator for &'c mut dyn BlockGenerator {

    fn write_range_assign(&mut self, target_ty: OutType, target: OutExpression, source_ty: OutType, source: OutExpression, len: OutExpression) -> OutResult {
        (**self).write_range_assign(target_ty, target, source_ty, source, len)
    }
    
    fn write_value_assign(&mut self, ty: OutType, assignee: OutExpression, val: OutExpression) -> OutResult {
        (**self).write_value_assign(ty, assignee, val)
    }

    fn write_assert(&mut self, value: OutExpression) -> OutResult {
        (**self).write_assert(value)
    }

    fn write_variable_declaration(&mut self, name: String, ty: OutType, value: Option<OutExpression>) -> OutResult {
        (**self).write_variable_declaration(name, ty, value)
    }

    fn write_return(&mut self, value: Option<OutExpression>) -> OutResult {
        (**self).write_return(value)
    }
    
    fn write_expr_statement(&mut self, expr: OutExpression) -> OutResult {
        (**self).write_expr_statement(expr)
    }

    fn write_entry_assign(&mut self, ty: OutType, arr: OutExpression, index: OutExpression, val: OutExpression) -> OutResult {
        (**self).write_entry_assign(ty, arr, index, val)
    }

    fn write_integer_for<'b>(
        &mut self, 
        name: String,
        init: OutExpression,
        limit: OutExpression,
        increment: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        (**self).write_integer_for(name, init, limit, increment, body)
    }

    fn write_if<'b>(
        &mut self, 
        condition: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        (**self).write_if(condition, body)
    }
    
    fn write_block<'b>(
        &mut self, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        (**self).write_block(body)
    }

    fn write_while<'b>(
        &mut self, 
        condition: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        (**self).write_while(condition, body)
    }

    fn write_parallel_code<'b>(
        &mut self, 
        thread_count: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>, OutExpression) -> OutResult>,
        used_outer_vars: Vec<(OutType, String)>
    ) -> OutResult {
        (**self).write_parallel_code(thread_count, body, used_outer_vars)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutPrimitiveType {
    Int, Long, Float, Double, Bool, UInt, SizeT, Struct(String)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutStorage {
    Value, PtrHost, PtrDevice, SmartPtrHost, SmartPtrDevice
}

impl OutStorage {

    pub fn is_device(&self) -> bool {
        *self == OutStorage::PtrDevice || *self == OutStorage::SmartPtrDevice
    }

    pub fn is_owned(&self) -> bool {
        *self == OutStorage::SmartPtrHost || *self == OutStorage::SmartPtrDevice
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutType {
    pub base: OutPrimitiveType,
    pub storage: OutStorage,
    pub mutable: bool
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutExpression {
    Sum(Vec<OutExpression>),
    Prod(Vec<OutExpression>),
    Call(Box<OutExpression>, Vec<OutExpression>),
    Symbol(String),
    Literal(i64),
    ThreadIndex(usize),
    Allocate(OutType, Box<OutExpression>),
    BracketExpr(Box<OutExpression>),
    IndexRead(OutType, Box<OutExpression>, Box<OutExpression>),
    StructMember(Box<OutExpression>, String),
    StructLiteral(Vec<OutExpression>),
    StaticCast(OutType, Box<OutExpression>),
    IndexOffset(OutType, Box<OutExpression>, Box<OutExpression>),
    Nullptr
}

impl OutExpression {

    fn priority(&self) -> i64 {
        match self {
            OutExpression::Sum(_) => 0,
            OutExpression::Prod(_) => 1,
            OutExpression::Call(_, _) => i64::MAX,
            OutExpression::Symbol(_) => i64::MAX,
            OutExpression::Literal(_) => i64::MAX,
            OutExpression::ThreadIndex(_) => i64::MAX,
            OutExpression::Allocate(_, _) => i64::MAX,
            OutExpression::BracketExpr(_) => i64::MAX,
            OutExpression::IndexRead(_, _, _) => i64::MAX,
            OutExpression::StructMember(_, _) => i64::MAX,
            OutExpression::StructLiteral(_) => i64::MAX,
            OutExpression::StaticCast(_, _) => i64::MAX,
            OutExpression::IndexOffset(_, _, _) => 0,
            OutExpression::Nullptr => i64::MAX
        }
    }

    fn wrap_if_prio_leq(expr: OutExpression, prio: i64) -> OutExpression {
        if expr.priority() <= prio { 
            OutExpression::BracketExpr(Box::new(expr)) 
        } else { 
            expr 
        }
    }

    fn wrap_iter_if_prio_leq<I: Iterator<Item = OutExpression>>(it: I, prio: i64) -> impl Iterator<Item = OutExpression> {
        it.map(move |e| Self::wrap_if_prio_leq(e, prio))
    }

    pub fn sum<I: Iterator<Item = OutExpression>>(it: I) -> Self {
        OutExpression::Sum(
            Self::wrap_iter_if_prio_leq(it, 0).collect()
        )
    }

    pub fn prod<I: Iterator<Item = OutExpression>>(it: I) -> Self {
        OutExpression::Prod(
            Self::wrap_iter_if_prio_leq(it, 1).collect()
        )
    }

    pub fn index_offset(ty: OutType, arr: OutExpression, offset: OutExpression) -> Self {
        OutExpression::IndexOffset(
            ty,
            Box::new(Self::wrap_if_prio_leq(arr, 0)),
            Box::new(Self::wrap_if_prio_leq(offset, 0))
        )
    }
}