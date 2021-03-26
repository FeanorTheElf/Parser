
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    Int, Bool, Float
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticType {
    base: PrimitiveType,
    dims: usize
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewType {
    view_onto: StaticType
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