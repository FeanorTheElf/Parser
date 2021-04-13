
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
        return_type: OutType, 
        body: Box<dyn 'b + FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult;
}

pub trait BlockGenerator {

    fn write_scalar_assign(&mut self, assignee: OutExpression, value: OutExpression) -> OutResult;
    fn write_variable_declaration(&mut self, name: String, ty: OutType, value: Option<OutExpression>) -> OutResult;
    fn write_return(&mut self, value: Option<OutExpression>) -> OutResult;
    fn write_expr_statement(&mut self, expr: OutExpression) -> OutResult;
    fn write_index_assign(&mut self, ty: OutType, arr: OutExpression, index: OutExpression, val: OutExpression) -> OutResult;

    fn write_if<'b>(
        &mut self, 
        condition: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult;
    
    fn write_while<'b>(
        &mut self, 
        condition: OutExpression, 
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

    fn write_scalar_assign(&mut self, assignee: OutExpression, value: OutExpression) -> OutResult {
        (**self).write_scalar_assign(assignee, value)
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

    fn write_index_assign(&mut self, ty: OutType, arr: OutExpression, index: OutExpression, val: OutExpression) -> OutResult {
        (**self).write_index_assign(ty, arr, index, val)
    }

    fn write_if<'b>(
        &mut self, 
        condition: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        (**self).write_if(condition, body)
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
    Int, Long, Float, Double, Bool, UInt
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutStorage {
    Value, PtrHost, PtrDevice, SmartPtrHost, SmartPtrDevice
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutType {
    pub base: OutPrimitiveType,
    pub storage: OutStorage,
    pub mutable: bool
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutExpression {
    Sum(Vec<Box<OutExpression>>),
    Prod(Vec<Box<OutExpression>>),
    Call(Box<OutExpression>, Vec<Box<OutExpression>>),
    Symbol(String),
    Literal(i64),
    ThreadIndex(usize),
    Allocate(OutType, Box<OutExpression>),
    BracketExpr(Box<OutExpression>),
    IndexRead(OutType, Box<OutExpression>, Box<OutExpression>)
}