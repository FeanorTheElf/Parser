use super::error::*;
use super::position::TextPosition;
use super::super::util::dynamic::{DynEq, Dynamic};
use super::super::util::dyn_lifetime::*;
use std::cell::RefCell;
use std::any::Any;

pub type TypePtr = DynRef<RefCell<Type>>;

#[derive(Debug)]
pub struct TypeVec {
    types: DynRefVec<RefCell<Type>>,
    array_types: Vec<DynRef<RefCell<Type>>>
}

impl TypeVec {
    pub fn new() -> Self {
        TypeVec {
            types: DynRefVec::new(),
            array_types: Vec::new()
        }
    }

    pub fn get_array_type(&mut self, base: PrimitiveType, dimension_count: usize) -> TypePtr {
        while self.array_types.len() <= dimension_count {
            let type_ref = self.types.push(RefCell::from(Type::Array(ArrayType {
                base: base,
                dimension: self.array_types.len()
            })));
            self.array_types.push(type_ref);
        }
        return self.array_types[dimension_count];
    }

    pub fn get_view_type(&mut self, base: PrimitiveType, dimension_count: usize) -> TypePtr {
        self.types.push(RefCell::from(Type::View(ViewType {
            base : ArrayType {
                base: base,
                dimension: dimension_count
            },
            concrete: None
        })))
    }

    pub fn get_function_type(&mut self, params: Vec<TypePtr>, return_type: Option<TypePtr>) -> TypePtr {
        self.types.push(RefCell::from(Type::Function(FunctionType {
            param_types: params,
            return_type: return_type
        })))
    }

    pub fn get_primitive_type(&mut self, ty: PrimitiveType) -> DynRef<RefCell<Type>> {
        return self.get_array_type(ty, 0);
    }

    pub fn get_lifetime<'a>(&'a self) -> Lifetime<'a> {
        self.types.get_lifetime()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PrimitiveType {
    Int, Float
}

pub trait ConcreteView : std::fmt::Debug + Any + DynEq + Dynamic {
    fn clone(&self) -> Box<dyn ConcreteView>;
}

impl Clone for Box<dyn ConcreteView> {
    fn clone(&self) -> Box<dyn ConcreteView> {
        self.clone()
    }
}

impl PartialEq for dyn ConcreteView {
    fn eq(&self, rhs: &dyn ConcreteView) -> bool {
        self.dyn_eq(rhs.dynamic())
    }
}

impl Eq for dyn ConcreteView {}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    TestType,
    JumpLabel,
    Array(ArrayType),
    Function(FunctionType),
    View(ViewType),
}

impl std::fmt::Display for Type {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Type::TestType => write!(out, "TestType"),
            Type::JumpLabel => write!(out, "JumpLabel"),
            Type::Array(ty) => write!(out, "{}", ty),
            Type::Function(ty) => write!(out, "{}", ty),
            Type::View(ty) => write!(out, "{}", ty),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionType {
    pub param_types: Vec<DynRef<RefCell<Type>>>,
    pub return_type: Option<DynRef<RefCell<Type>>>
}

impl std::fmt::Display for FunctionType {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(out, "fn(")?;
        for param in &self.param_types {
            write!(out, "{:?}, ", param)?;
        }
        write!(out, ")")?;
        if let Some(ret) = &self.return_type {
            write!(out, ": {:?}", ret)?;
        }
        return Ok(());
    }
}

impl FunctionType {
    fn write(&self, prog_lifetime: Lifetime, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(out, "fn(")?;
        for param in &self.param_types {
            write!(out, "{}, ", prog_lifetime.cast(*param).borrow())?;
        }
        write!(out, ")")?;
        if let Some(ret) = &self.return_type {
            write!(out, ": {}", prog_lifetime.cast(*ret).borrow())?;
        }
        return Ok(());
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ArrayType {
    pub base: PrimitiveType,
    pub dimension: usize
}

impl std::fmt::Display for ArrayType {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(out, "{:?}[{}]", self.base, ",".to_owned().repeat(self.dimension))
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ViewType {
    pub base: ArrayType,
    pub concrete: Option<Box<dyn ConcreteView>>
}

impl std::fmt::Display for ViewType {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(out, "&{}", self.base)
    }
}

impl Type {

    pub fn without_view(self) -> Type {
        match self {
            Type::View(view) => Type::Array(view.base),
            ty => ty
        }
    }

    pub fn is_view(&self) -> bool {
        match self {
            Type::View(view) => true,
            _ => false
        }
    }

    pub fn expect_callable(
        &self,
        pos: &TextPosition,
    ) -> Result<&FunctionType, CompileError> {
        match self {
            Type::Function(function_type) => Ok(function_type),
            ty => Err(CompileError::new(
                pos,
                format!("Expression of type {} is not callable", ty),
                ErrorType::TypeError,
            )),
        }
    }

    pub fn expect_array(
        &self,
        pos: &TextPosition
    ) -> Result<&ArrayType, CompileError> {
        match self {
            Type::Array(array_type) => Ok(array_type),
            ty => Err(CompileError::new(
                pos,
                format!("Expression of type {} is not an owned array", ty),
                ErrorType::TypeError,
            )),
        }
    }

    pub fn expect_indexable(
        &self,
        pos: &TextPosition
    ) -> Result<&ArrayType, CompileError> {
        match self {
            Type::Array(array_type) => Ok(array_type),
            Type::View(view_type) => Ok(&view_type.base),
            ty => Err(CompileError::new(
                pos,
                format!("Expression of type {} is not indexable", ty),
                ErrorType::TypeError,
            )),
        }
    }

    
    pub fn reference_view(self, pos: &TextPosition) -> Result<ViewType, CompileError> {
        match self {
            Type::Array(array_type) => Ok(ViewType {
                base: array_type,
                concrete: None
            }),
            Type::View(view) => Ok(view),
            ty => Err(CompileError::new(
                pos,
                format!("Views on type {} do not exist", ty),
                ErrorType::TypeError,
            ))
        }
    }
}