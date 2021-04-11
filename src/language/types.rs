use super::super::util::dynamic::DynEq;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    Int, Bool, Float
}

impl PrimitiveType {

    pub const fn array(self, dims: usize) -> Type {
        Type::array_type(self, dims)
    }

    pub const fn scalar(self) -> Type {
        self.array(0)
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
    pub dims: usize
}

impl StaticType {

    pub fn is_scalar(&self) -> bool {
        self.dims == 0
    }

    pub fn as_scalar(&self) -> Option<&PrimitiveType> {
        if self.is_scalar() {
            return Some(&self.base);
        } else {
            return None;
        }
    }
}

pub const SCALAR_INT: Type = PrimitiveType::Int.scalar();
pub const SCALAR_BOOL: Type = PrimitiveType::Bool.scalar();
pub const SCALAR_FLOAT: Type = PrimitiveType::Float.scalar();

pub trait ViewFuncs: std::fmt::Debug + std::any::Any + DynEq {

}

dynamic_trait_cloneable!{ View: ViewFuncs; ViewDynCastable }

#[derive(Debug)]
pub struct ViewType {
    pub view_onto: StaticType,
    pub concrete_view: Option<Box<dyn View>>,
    mutable: bool
}

impl ViewType {
    
    pub fn is_mutable(&self) -> bool {
        self.mutable
    }
}

impl Clone for ViewType {

    fn clone(&self) -> ViewType {
        ViewType {
            view_onto: self.view_onto.clone(),
            concrete_view: self.concrete_view.as_ref().map(|x| x.dyn_clone()),
            mutable: self.mutable
        }
    }
}

impl PartialEq for ViewType {

    fn eq(&self, rhs: &ViewType) -> bool {
        if self.view_onto != rhs.view_onto {
            return false;
        }
        if self.concrete_view.is_some() != rhs.concrete_view.is_some() {
            return false;
        }
        if self.concrete_view.is_some() && 
            !self.concrete_view.as_ref().unwrap().dyn_eq(rhs.concrete_view.as_ref().unwrap().any()) 
        {
            return false;
        }
        return true;
    }
}

impl Eq for ViewType {}

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

impl Type {

    pub const fn array_type(base: PrimitiveType, dims: usize) -> Type {
        Type::Static(StaticType {
            base, dims
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

    pub const fn scalar_type(primitive: PrimitiveType) -> Type {
        Self::array_type(primitive, 0)
    }

    pub fn with_concrete_view<T: View>(self, concrete_view: T, mutable: bool) -> Type {
        match self {
            Type::Static(static_type) => Type::View(ViewType {
                view_onto: static_type,
                concrete_view: Some(Box::new(concrete_view)),
                mutable: mutable
            }),
            Type::View(_) => panic!("with_concrete_view() will not override existing concrete_view attribute"),
            Type::Function(_) => unimplemented!("TODO: what are views on functions?")
        }
    }

    pub fn with_view(self, mutable: bool) -> Type {
        match self {
            Type::Static(static_type) => Type::View(ViewType {
                view_onto: static_type,
                concrete_view: None,
                mutable: mutable
            }),
            Type::View(view_type) => Type::View(view_type),
            Type::Function(_) => unimplemented!("TODO: what are views on functions?")
        }
    }

    pub fn as_view(&self) -> Option<&ViewType> {
        match self {
            Type::View(v) => Some(v),
            _ => None
        }
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

    pub fn as_scalar(&self) -> Option<&PrimitiveType> {
        self.as_static().and_then(StaticType::as_scalar)
    }

    pub fn is_implicitly_convertable(&self, target: &Type) -> bool {
        *self == *target || (*self == SCALAR_INT && *target == SCALAR_FLOAT) 
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

impl PartialEq<PrimitiveType> for Type {

    fn eq(&self, rhs: &PrimitiveType) -> bool {
        self.as_scalar().map(|s| s == rhs).unwrap_or(false)
    }
}