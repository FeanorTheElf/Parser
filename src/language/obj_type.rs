use super::super::parser::ast::TypeDecl;

pub enum BasicType {
    Int
}

pub enum Type {
    Basic(BasicType),
    Array(BasicType, u32),
    Function(Vec<Type>, Option<Box<Type>>)
}

impl Type {
    pub fn from(t: &TypeDecl) -> Option<Box<Type>> {
        match t {
            TypeDecl::Arr(ref _annotation, ref basic, dims) => {
                Some(Box::new(Type::Array(BasicType::Int, *dims as u32)))
            },
            TypeDecl::Void(ref _annotation) => {
                None
            }
        }
    }
}