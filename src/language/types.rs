
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    Int, Bool, Float
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

pub const SCALAR_INT: Type = Type::Static(StaticType { base: PrimitiveType::Int, dims: 0 });
pub const SCALAR_BOOL: Type = Type::Static(StaticType { base: PrimitiveType::Bool, dims: 0 });
pub const SCALAR_FLOAT: Type = Type::Static(StaticType { base: PrimitiveType::Float, dims: 0 });

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewType {
    pub view_onto: StaticType
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Static(StaticType),
    View(ViewType)
}

impl Type {

    pub fn scalar_type(primitive: PrimitiveType) -> Type {
        Type::Static(StaticType {
            base: primitive,
            dims: 0
        })
    }

    pub fn as_view(&self) -> Option<&ViewType> {
        match self {
            Type::Static(_) => None,
            Type::View(v) => Some(v)
        }
    }

    pub fn as_view_mut(&mut self) -> Option<&mut ViewType> {
        match self {
            Type::Static(_) => None,
            Type::View(v) => Some(v)
        }
    }

    pub fn as_static(&self) -> Option<&StaticType> {
        match self {
            Type::Static(s) => Some(s),
            Type::View(_) => None
        }
    }

    pub fn as_static_mut(&mut self) -> Option<&mut StaticType> {
        match self {
            Type::Static(s) => Some(s),
            Type::View(_) => None
        }
    }
}