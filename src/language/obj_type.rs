
pub enum BasicType {
    Int
}

pub enum Type {
    Basic(BasicType),
    Array(BasicType, u32),
    Function(Vec<Type>, Box<Type>)
}