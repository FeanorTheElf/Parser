use super::super::util::dynamic::DynEq;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    Int, Bool, Float
}

impl PrimitiveType {

    pub const fn array(self, dims: usize, mutable: bool) -> Type {
        Type::array_type(self, dims, mutable)
    }

    pub const fn scalar(self, mutable: bool) -> Type {
        self.array(0, mutable)
    }
}

impl std::fmt::Display for PrimitiveType {

    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PrimitiveType::Int => write!(f, "int"),
            PrimitiveType::Bool => write!(f, "bool"),
            PrimitiveType::Float => write!(f, "float")
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticType {
    pub base: PrimitiveType,
    pub dims: usize,
    pub mutable: bool
}

impl StaticType {

    pub fn is_scalar(&self) -> bool {
        self.dims == 0
    }

    pub fn get_base(&self) -> PrimitiveType {
        self.base
    }

    pub fn is_mutable(&self) -> bool {
        self.mutable
    }
}

pub trait ViewFuncs: std::fmt::Debug + std::any::Any + DynEq {

}

dynamic_trait_cloneable!{ View: ViewFuncs; ViewDynCastable }

impl Clone for Box<dyn View> {

    fn clone(&self) -> Box<dyn View> {
        self.dyn_clone()
    }
}

impl PartialEq for &dyn View {

    fn eq(&self, rhs: &&dyn View) -> bool {
        (*self).dyn_eq((*rhs).any())
    }
}

impl Eq for &dyn View {}

impl PartialEq for Box<dyn View> {

    fn eq(&self, rhs: &Box<dyn View>) -> bool {
        self.dyn_eq(&*rhs)
    }
}

impl Eq for Box<dyn View> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewType {
    pub view_onto: StaticType,
    pub concrete_view: Option<Box<dyn View>>
}

impl ViewType {

    pub fn get_concrete(&self) -> &dyn View {
        &**self.concrete_view.as_ref().expect("concrete view not set")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType {
    parameters: Vec<Box<Type>>,
    return_type: Option<Box<Type>>
}

impl FunctionType {

    pub fn parameter_types<'a>(&'a self) -> impl 'a + Iterator<Item = &'a Type> {
        self.parameters.iter().map(|x| &**x)
    }

    pub fn return_type(&self) -> Option<&Type> {
        self.return_type.as_ref().map(|x| &**x)
    }

    pub fn is_void(&self) -> bool {
        self.return_type().is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Static(StaticType),
    View(ViewType),
    Function(FunctionType)
}

pub const SCALAR_INT: Type = PrimitiveType::Int.scalar(false);
pub const SCALAR_FLOAT: Type = PrimitiveType::Float.scalar(false);
pub const SCALAR_BOOL: Type = PrimitiveType::Bool.scalar(false);

impl Type {

    pub const fn array_type(base: PrimitiveType, dims: usize, mutable: bool) -> Type {
        Type::Static(StaticType {
            base, dims, mutable
        })
    }

    pub fn function_type<I>(parameter_types: I, return_type: Option<Type>) -> Type 
        where I: Iterator<Item = Type>
    {
        Type::Function(FunctionType {
            parameters: parameter_types.map(Box::new).collect(),
            return_type: return_type.map(Box::new)
        })
    }

    pub const fn scalar_type(primitive: PrimitiveType, mutable: bool) -> Type {
        Self::array_type(primitive, 0, mutable)
    }

    pub fn with_concrete_view<T: View>(self, concrete_view: T) -> Type {
        self.with_concrete_view_dyn(Box::new(concrete_view))
    }

    pub fn with_concrete_view_dyn(self, concrete_view: Box<dyn View>) -> Type {
        match self {
            Type::Static(static_type) => Type::View(ViewType {
                view_onto: static_type,
                concrete_view: Some(concrete_view)
            }),
            Type::View(_) => panic!("with_concrete_view() will not override existing concrete_view attribute"),
            Type::Function(_) => unimplemented!("TODO: what are views on functions?")
        }
    }

    pub fn with_view(self) -> Type {
        match self {
            Type::Static(static_type) => Type::View(ViewType {
                view_onto: static_type,
                concrete_view: None
            }),
            Type::View(view_type) => Type::View(view_type),
            Type::Function(_) => unimplemented!("TODO: what are views on functions?")
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.as_static().map(|s| s.is_scalar()).unwrap_or(false)
    }

    pub fn as_view(&self) -> Option<&ViewType> {
        match self {
            Type::View(v) => Some(v),
            _ => None
        }
    }

    pub fn is_view(&self) -> bool {
        self.as_view().is_some()
    }

    pub fn as_view_mut(&mut self) -> Option<&mut ViewType> {
        match self {
            Type::View(v) => Some(v),
            _ => None
        }
    }

    pub fn as_static(&self) -> Option<&StaticType> {
        match self {
            Type::Static(s) => Some(s),
            _ => None
        }
    }

    pub fn as_static_mut(&mut self) -> Option<&mut StaticType> {
        match self {
            Type::Static(s) => Some(s),
            _ => None
        }
    }

    pub fn as_function(&self) -> Option<&FunctionType> {
        match self {
            Type::Function(s) => Some(s),
            _ => None
        }
    }

    pub fn as_function_mut(&mut self) -> Option<&mut FunctionType> {
        match self {
            Type::Function(s) => Some(s),
            _ => None
        }
    }

    ///
    /// Returns whether a value of this type is implicitly convertable into
    /// a value of the target type.
    /// Currently, this is only the case for identical types and extending
    /// scalar type conversions.
    /// 
    pub fn is_implicitly_convertable(&self, target: &Type) -> bool {
        (self.is_scalar() && *self == *target) || 
            (self.is_scalar_of(PrimitiveType::Int) && target.is_scalar_of(PrimitiveType::Float))
    }

    ///
    /// Returns whether a view that has the target type can be taken of an
    /// object of this type
    /// 
    pub fn is_viewable_as(&self, target: &Type) -> bool {
        assert!(target.is_view());
        match (self, target) {
            (Type::Static(from), Type::View(to)) => {
                from.base == to.view_onto.base &&
                    from.dims == to.view_onto.dims &&
                    (from.mutable || !to.view_onto.mutable)
            },
            (Type::View(from), Type::View(to)) => {
                from.view_onto.base == to.view_onto.base &&
                    from.view_onto.dims == to.view_onto.dims &&
                    (from.view_onto.mutable || !to.view_onto.mutable)
            },
            (_, _) => false
        }
    }

    pub fn is_scalar_of(&self, rhs: PrimitiveType) -> bool {
        match self {
            Type::Static(s) if s.is_scalar() => s.get_base() == rhs,
            _ => false
        }
    }
}

pub type VoidableType = Option<Type>;

impl PartialEq<StaticType> for Type {

    fn eq(&self, rhs: &StaticType) -> bool {
        self.as_static().map(|s| s == rhs).unwrap_or(false)
    }
}

impl PartialEq<ViewType> for Type {

    fn eq(&self, rhs: &ViewType) -> bool {
        self.as_view().map(|s| s == rhs).unwrap_or(false)
    }
}
