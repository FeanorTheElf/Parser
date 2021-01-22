use super::error::*;
use super::position::{TextPosition, NONEXISTING};
use super::super::util::dynamic::DynEq;
use super::super::util::dyn_lifetime::*;
use std::cell::{RefCell, Ref};

///
/// On the semantic of pointer equality of types:
///  - There are completely code determined types (i.e. types whose string in the source code determines them completely). For these types
///    (mostly arrays), there is just one instance and all pointers point to this instance
///  - For types where this is not the case (mostly views), use the same pointer in two locations, if all parts of the type are necessarily
///    equal for both locations (e.g. because it is the type of the same variable); If this is not the case (e.g. a variable containing a copy
///    of another variable), copy the type and use a pointers to each copy 
/// 
pub type TypePtr = DynRef<Type>;

pub const HASH_REFERENCEVIEW: u32 = 0;
pub const HASH_ZEROVIEW: u32 = 1;
pub const HASH_INDEXVIEW: u32 = 2;
pub const HASH_COMPOSEDVIEW: u32 = 3;
pub const HASH_COMPLETEINDEXVIEW: u32 = 4;
pub const HASH_TEMPLATE: u32 = 5;

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
    Int, Float, Bool
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
        (HASH_TEMPLATE << 24) | (self.id as u32 & 0xFFFFFF)
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

#[derive(Debug, PartialEq, Eq)]
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

#[derive(Debug, PartialEq, Eq)]
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

    pub fn clone(self_ptr: TypePtr, types: &mut TypeVec) -> TypePtr {
        let mut param_types = Vec::new();
        let param_ptrs = self_ptr.deref(types.get_lifetime()).expect_callable(&NONEXISTING).internal_error().param_types.clone();
        for param in param_ptrs {
            param_types.push(Type::clone(param, types));
        }
        let return_ptr = self_ptr.deref(types.get_lifetime()).expect_callable(&NONEXISTING).internal_error().return_type;
        let return_type = Type::clone_voidable(return_ptr, types);
        return types.get_function_type(param_types, return_type);
    }

    pub fn replace_templated_view_parts(self_ptr: TypePtr, template: Template, target: &dyn ConcreteView, type_lifetime: &mut LifetimeMut) {
        let n = self_ptr.deref(type_lifetime.as_const()).expect_callable(&NONEXISTING).internal_error().param_types.len();
        for i in 0..n {
            let ptr = self_ptr.deref(type_lifetime.as_const()).expect_callable(&NONEXISTING).internal_error().param_types[i];
            Type::replace_templated_view_parts(ptr, template, target, type_lifetime);
        }
        if let VoidableTypePtr::Some(ptr) = self_ptr.deref(type_lifetime.as_const()).expect_callable(&NONEXISTING).internal_error().return_type {
            Type::replace_templated_view_parts(ptr, template, target, type_lifetime);
        }
    }

    pub fn fill_concrete_views_with_template(self_ptr: TypePtr, type_lifetime: &mut LifetimeMut) {
        let n = self_ptr.deref(type_lifetime.as_const()).expect_callable(&NONEXISTING).internal_error().param_types.len();
        let mut id = 0;
        for i in 0..n {
            let param_type_ptr = self_ptr.deref(type_lifetime.as_const()).expect_callable(&NONEXISTING).internal_error().param_types[i];
            if let Type::View(view) = param_type_ptr.deref_mut(type_lifetime) {
                assert!(view.concrete.is_none());
                view.concrete = Some(Box::new(Template::new(id)));
                id += 1;
            }
        }
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

    pub fn replace_templated_view_parts(self_ptr: TypePtr, template: Template, target: &dyn ConcreteView, type_lifetime: &mut LifetimeMut) {
        match self_ptr.deref_mut(type_lifetime) {
            Type::View(view) =>
                take_mut::take(view.concrete.as_mut().unwrap(), |concrete| {
                    concrete.replace_templated(template, target)
                }),
            _ => panic!("ViewType::replace_templated_view_parts() called for non-view type")
        }
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

    pub fn expect_arithmetic(&self, pos: &TextPosition) -> Result<PrimitiveType, CompileError> {
        if let Type::Array(arr) = self {
            if arr.dimension == 0 {
                if arr.base == PrimitiveType::Int || arr.base == PrimitiveType::Float {
                    return Ok(arr.base)
                }
            }
        }
        return Err(CompileError::new(
            pos, 
            format!("Cannot perform arithmetic operations on type {}", self), 
            ErrorType::TypeError
        ));
    }
    
    pub fn expect_scalar(&self, pos: &TextPosition) -> Result<PrimitiveType, CompileError> {
        if let Type::Array(arr) = self {
            if arr.dimension == 0 {
                return Ok(arr.base)
            }
        }
        return Err(CompileError::new(
            pos, 
            format!("Expected a scalar type, got {}", self), 
            ErrorType::TypeError
        ));
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

    ///
    /// Returns a pointer to a type that is equivalent to the given one. This pointer may
    /// or may not be the same as the given one, depending on the semantics for the given type.
    ///  
    /// In particular, arrays will always have the same pointer, but view types will use different
    /// pointers. Therefore, the given and the returned type will behave in the same way as two types
    /// that are obtained by parsing two times the same expression and calculating its type.
    /// 
    pub fn clone(self_ptr: TypePtr, types: &mut TypeVec) -> TypePtr {
        let result = match self_ptr.deref(types.get_lifetime()) {
            Type::Function(_) => FunctionType::clone(self_ptr, types),
            Type::Array(arr) => {
                let base = arr.base;
                let dimension = arr.dimension;
                let mutable = arr.mutable;
                types.get_array_type(base, dimension, mutable)
            },
            Type::View(view) => {
                let base = view.base.base;
                let dimension = view.base.dimension;
                let mutable = view.base.mutable;
                let concrete = view.concrete.clone();
                types.get_view_type(base, dimension, mutable, concrete)
            },
            Type::JumpLabel => types.get_jump_label_type(),
            Type::TestType => types.get_test_type_type()
        };
        assert!((result == self_ptr) == self_ptr.deref(types.get_lifetime()).is_code_determined());
        return result;
    }

    ///
    /// Returns whether this type is already completely determined by their string representation in code
    /// (i.e. arrays), as opposed to e.g. views, for which there is some type information that is not represented
    /// in code, but infered by type analysis during compile time.
    /// 
    /// These also have different pointer equality semantics, see `TypePtr`
    /// 
    pub fn is_code_determined(&self) -> bool {
        match self {
            Type::Function(_) => false,
            Type::Array(_) => true,
            Type::View(_) => false,
            Type::JumpLabel => true,
            Type::TestType => true
        }
    }

    pub fn clone_voidable(self_ptr: VoidableTypePtr, types: &mut TypeVec) -> VoidableTypePtr {
        match self_ptr {
            VoidableTypePtr::Void => VoidableTypePtr::Void,
            VoidableTypePtr::Some(ptr) => VoidableTypePtr::Some(Type::clone(ptr, types))
        }
    }

    pub fn replace_templated_view_parts(self_ptr: TypePtr, template: Template, target: &dyn ConcreteView, type_lifetime: &mut LifetimeMut) {
        match self_ptr.deref(type_lifetime.as_const()) {
            Type::View(_) => ViewType::replace_templated_view_parts(self_ptr, template, target, type_lifetime),
            Type::Function(_) => FunctionType::replace_templated_view_parts(self_ptr, template, target, type_lifetime),
            ty => assert!(ty.is_code_determined()) 
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

    /// Whether the content of an object of this type can be copied into a variable of the target type. Use this
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