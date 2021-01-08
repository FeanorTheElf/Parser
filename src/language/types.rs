use super::error::*;
use super::position::TextPosition;
use super::super::util::dynamic::DynEq;
use super::super::util::dyn_lifetime::*;
use std::cell::{RefCell, Ref};

pub type TypePtr = DynRef<Type>;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VoidableTypePtr {
    Some(TypePtr),
    Void
}

impl VoidableTypePtr {
    pub fn is_void(&self) -> bool {
        match self {
            VoidableTypePtr::Some(_) => false,
            VoidableTypePtr::Void => true
        }
    }

    pub fn unwrap(&self) -> TypePtr {
        match self {
            VoidableTypePtr::Some(val) => *val,
            VoidableTypePtr::Void => panic!("Called unwrap() on a type with value void")
        }
    }

    pub fn map<T, F>(&self, f: F) -> Option<T> 
        where F: FnOnce(TypePtr) -> T
    {
        match self {
            VoidableTypePtr::Some(val) => Some(f(*val)),
            VoidableTypePtr::Void => None
        }
    } 

    pub fn expect_nonvoid(&self, pos: &TextPosition) -> Result<TypePtr, CompileError> {
        match self {
            VoidableTypePtr::Some(val) => Ok(*val),
            VoidableTypePtr::Void => Err(CompileError::new(pos, format!("Expected type, got void"), ErrorType::TypeError))
        }
    }
}

#[derive(Debug)]
pub struct TypeVec {
    types: DynRefVec<Type>,
    array_types: Vec<TypePtr>,
    jump_label_type: TypePtr,
    test_type_type: TypePtr
}

impl TypeVec {
    pub fn new() -> Self {
        let mut types = DynRefVec::new();
        let jump_label_type = types.push(Type::JumpLabel);
        let test_type_type = types.push(Type::TestType);
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
            let type_ref = self.types.push(Type::Array(ArrayType {
                base: base,
                dimension: self.array_types.len(),
                mutable: mutable
            }));
            self.array_types.push(type_ref);
        }
        return self.array_types[dimension_count];
    }

    pub fn get_view_type(&mut self, base: PrimitiveType, dimension_count: usize, mutable: bool, concrete_view: Option<Box<dyn ConcreteView>>) -> TypePtr {
        self.types.push(Type::View(ViewType {
            base : ArrayType {
                base: base,
                dimension: dimension_count,
                mutable: mutable
            },
            concrete: concrete_view
        }))
    }

    pub fn get_generic_view_type(&mut self, base: PrimitiveType, dimension_count: usize, mutable: bool) -> TypePtr {
        self.get_view_type(base, dimension_count, mutable, None)
    }

    pub fn get_function_type(&mut self, params: Vec<TypePtr>, return_type: VoidableTypePtr) -> TypePtr {
        self.types.push(Type::Function(FunctionType {
            param_types: params,
            return_type: return_type
        }))
    }

    pub fn get_jump_label_type(&self) -> TypePtr {
        self.jump_label_type
    }

    pub fn get_test_type_type(&self) -> TypePtr {
        self.test_type_type
    }

    pub fn get_primitive_type(&mut self, ty: PrimitiveType, mutable: bool) -> TypePtr {
        return self.get_array_type(ty, 0, mutable);
    }

    pub fn get_lifetime<'a>(&'a self) -> Lifetime<'a> {
        self.types.get_lifetime()
    }

    pub fn get_lifetime_mut<'a>(&'a mut self) -> LifetimeMut<'a> {
        self.types.get_lifetime_mut()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PrimitiveType {
    Int, Float
}

dynamic_trait_cloneable!{ ConcreteView: ConcreteViewFuncs; ConcreteViewDynCastable }

pub trait ConcreteViewFuncs : std::fmt::Debug + std::any::Any + DynEq {
    ///
    /// One or more alphanumeric characters that can be integrated e.g. into function names
    /// during monomorphization (to distinguish between different instantiations of the same
    /// function with different concrete views as parameter)
    /// 
    fn identifier(&self) -> String;
    fn hash(&self) -> u32;
    fn replace_templated(self: Box<Self>, value: Template, target: &dyn ConcreteView) -> Box<dyn ConcreteView>;
    fn contains_templated(&self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Template {
    id: usize
}

impl Template {
    pub fn new(id: usize) -> Template {
        Template {
            id: id
        }
    }
}

impl ConcreteViewFuncs for Template {
    fn identifier(&self) -> String {
        format!("t{}", self.id)
    }

    fn hash(&self) -> u32 {
        unimplemented!()
    }

    fn replace_templated(self: Box<Self>, value: Template, target: &dyn ConcreteView) -> Box<dyn ConcreteView> {
        if *self == value {
            target.dyn_clone()
        } else {
            self
        }
    }

    fn contains_templated(&self) -> bool {
        true
    }
}

impl std::hash::Hash for Template {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        <dyn ConcreteView as std::hash::Hash>::hash::<H>(self, hasher)
    }
}

impl std::cmp::PartialOrd for Template {
    fn partial_cmp(&self, rhs: &Template) -> Option<std::cmp::Ordering> {
        Some(self.cmp(rhs))
    }
}

impl std::cmp::Ord for Template {
    fn cmp(&self, rhs: &Template) -> std::cmp::Ordering {
        self.id.cmp(&rhs.id)
    }
}

impl ConcreteView for Template {}

impl Clone for Box<dyn ConcreteView> {
    fn clone(&self) -> Box<dyn ConcreteView> {
        self.dyn_clone()
    }
}

impl PartialEq for dyn ConcreteView {
    fn eq(&self, rhs: &dyn ConcreteView) -> bool {
        self.dyn_eq(rhs.any())
    }
}

impl Eq for dyn ConcreteView {}

impl std::hash::Hash for dyn ConcreteView {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        hasher.write_u32(self.hash())
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    TestType,
    JumpLabel,
    Array(ArrayType),
    Function(FunctionType),
    View(ViewType)
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
    pub param_types: Vec<TypePtr>,
    pub return_type: VoidableTypePtr
}

impl std::fmt::Display for FunctionType {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(out, "fn(")?;
        for param in &self.param_types {
            write!(out, "{:?}, ", param)?;
        }
        write!(out, ")")?;
        if let VoidableTypePtr::Some(ret) = &self.return_type {
            write!(out, ": {:?}", ret)?;
        }
        return Ok(());
    }
}

impl FunctionType {
    pub fn write(&self, prog_lifetime: Lifetime, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(out, "fn(")?;
        for param in &self.param_types {
            prog_lifetime.cast(*param).write(prog_lifetime, out)?;
            write!(out, ", ")?;
        }
        write!(out, ")")?;
        if let VoidableTypePtr::Some(ret) = &self.return_type {
            write!(out, ": ")?;
            prog_lifetime.cast(*ret).write(prog_lifetime, out)?;
        }
        return Ok(());
    }

    pub fn return_type<'a, 'b: 'a>(&'a self, prog_lifetime: Lifetime<'b>) -> Option<&'a Type> {
        self.return_type.map(|ty| prog_lifetime.cast(ty))
    }
    
    pub fn param_types<'a, 'b: 'a>(&'a self, prog_lifetime: Lifetime<'b>) -> impl 'a + Iterator<Item = &'a Type> {
        self.param_types.iter().map(move |p| prog_lifetime.cast(*p))
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoConcreteViewDataPresent;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ViewType {
    pub base: ArrayType,
    pub concrete: Option<Box<dyn ConcreteView>>
}

impl ViewType {
    pub fn get_concrete(&self) -> Result<&dyn ConcreteView, NoConcreteViewDataPresent> {
        self.concrete.as_ref().map(|c| &**c).ok_or(NoConcreteViewDataPresent)
    }
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

    pub fn expect_view(
        &self,
        pos: &TextPosition
    ) -> Result<&ViewType, CompileError> {
        match self {
            Type::View(view_type) => Ok(&view_type),
            ty => Err(CompileError::new(
                pos,
                format!("Expression of type {} is not a view", ty),
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

pub struct LifetimedType<'a> {
    lifetime: Lifetime<'a>,
    type_ptr: TypePtr
}

impl<'a> LifetimedType<'a> {
    pub fn bind(type_ptr: TypePtr, lifetime: Lifetime<'a>) -> LifetimedType<'a> {
        LifetimedType {
            type_ptr, lifetime
        }
    }
}

impl<'a> std::fmt::Display for LifetimedType<'a> {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.type_ptr.deref(self.lifetime) {
            Type::Function(function_type) => function_type.write(self.lifetime, out),
            ty => write!(out, "{}", ty)
        }
    }
}