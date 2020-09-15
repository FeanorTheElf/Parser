use super::error::*;
use super::position::TextPosition;
use super::super::util::dynamic::{DynEq, Dynamic};
use super::super::util::dyn_lifetime::*;
use std::cell::{RefCell, Ref};
use std::any::Any;

pub type TypePtr = DynRef<RefCell<Type>>;

#[derive(Debug)]
pub struct TypeVec {
    types: DynRefVec<RefCell<Type>>,
    array_types: Vec<TypePtr>,
    jump_label_type: TypePtr,
    test_type_type: TypePtr
}

impl TypeVec {
    pub fn new() -> Self {
        let mut types = DynRefVec::new();
        let jump_label_type = types.push_from(Type::JumpLabel);
        let test_type_type = types.push_from(Type::TestType);
        let array_types = Vec::new();
        TypeVec {
            types,
            array_types,
            jump_label_type,
            test_type_type
        }
    }

    pub fn get_array_type(&mut self, base: PrimitiveType, dimension_count: usize, mutable: bool) -> TypePtr {
        while self.array_types.len() <= dimension_count {
            let type_ref = self.types.push(RefCell::from(Type::Array(ArrayType {
                base: base,
                dimension: self.array_types.len(),
                mutable: mutable
            })));
            self.array_types.push(type_ref);
        }
        return self.array_types[dimension_count];
    }

    pub fn get_view_type(&mut self, base: PrimitiveType, dimension_count: usize, mutable: bool) -> TypePtr {
        self.types.push(RefCell::from(Type::View(ViewType {
            base : ArrayType {
                base: base,
                dimension: dimension_count,
                mutable: mutable
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

    pub fn get_jump_label_type(&self) -> TypePtr {
        self.jump_label_type
    }

    pub fn get_test_type_type(&self) -> TypePtr {
        self.test_type_type
    }

    pub fn get_primitive_type(&mut self, ty: PrimitiveType, mutable: bool) -> DynRef<RefCell<Type>> {
        return self.get_array_type(ty, 0, mutable);
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
    ///
    /// One or more alphanumeric characters that can be integrated e.g. into function names
    /// during monomorphization (to distinguish between different instantiations of the same
    /// function with different concrete views as parameter)
    /// 
    fn identifier(&self) -> String;
}

impl Clone for Box<dyn ConcreteView> {
    fn clone(&self) -> Box<dyn ConcreteView> {
        ConcreteView::clone(&**self)
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

impl Type {
    pub fn write(&self, prog_lifetime: Lifetime, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Type::Function(ty) => ty.write(prog_lifetime, out),
            ty => write!(out, "{}", ty)
        }
    }
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
    pub fn write(&self, prog_lifetime: Lifetime, out: &mut std::fmt::Formatter) -> std::fmt::Result {
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

    pub fn return_type<'a, 'b: 'a>(&'a self, prog_lifetime: Lifetime<'b>) -> Option<Ref<'a, Type>> {
        self.return_type.map(|ty| prog_lifetime.cast(ty).borrow())
    }
    
    pub fn param_types<'a, 'b: 'a>(&'a self, prog_lifetime: Lifetime<'b>) -> impl 'a + Iterator<Item = Ref<'a, Type>> {
        self.param_types.iter().map(move |p| prog_lifetime.cast(*p).borrow())
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ArrayType {
    pub base: PrimitiveType,
    pub dimension: usize,
    pub mutable: bool
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
            Type::View(_) => true,
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

    /// Whether objects of this type are implicitly convertable into objects of the target type. Use this
    /// to check if given parameters are valid
    pub fn is_implicitly_convertable(&self, target: &Type, _prog_lifetime: Lifetime) -> bool {
        match self {
            Type::Array(self_type) => match target {
                Type::Array(target_type) => self_type.base == target_type.base && self_type.dimension == target_type.dimension,
                Type::View(target_type) => self_type.base == target_type.base.base && self_type.dimension == target_type.base.dimension && !(!self_type.mutable && target_type.base.mutable),
                _ => false
            },
            Type::View(self_type) => match target {
                Type::View(target_type) => self_type.base.base == target_type.base.base && self_type.base.dimension == target_type.base.dimension && !(!self_type.base.mutable && target_type.base.mutable),
                _ => false
            },
            Type::Function(_) => unimplemented!(),
            Type::TestType => *target == Type::TestType,
            Type::JumpLabel => *target == Type::JumpLabel
        }
    }

    /// Whether the content of an object of this type can be copied into an object of the target type. Use this
    /// to check if assignments are valid
    pub fn is_copyable_to(&self, target: &Type, _prog_lifetime: Lifetime) -> bool {
        match target {
            Type::Array(target_type) => (match self {
                Type::Array(self_type) => self_type.base == target_type.base && self_type.dimension == target_type.dimension,
                Type::View(self_type) => self_type.base.base == target_type.base && self_type.base.dimension == target_type.dimension,
                _ => false
            } && target_type.mutable),
            Type::View(target_type) => (match self {
                Type::Array(self_type) => self_type.base == target_type.base.base && self_type.dimension == target_type.base.dimension,
                Type::View(self_type) => self_type.base.base == target_type.base.base && self_type.base.dimension == target_type.base.dimension, 
                _ => false
            } && target_type.base.mutable),
            Type::Function(_) => unimplemented!(),
            Type::TestType => *self == Type::TestType,
            Type::JumpLabel => *self == Type::JumpLabel
        }
    }
}