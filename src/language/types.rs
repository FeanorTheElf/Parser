
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PrimitiveType {
    Int,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    TestType,
    JumpLabel,
    Primitive(PrimitiveType),
    Array(PrimitiveType, u32),
    Function(Vec<Box<Type>>, Option<Box<Type>>),
    View(Box<Type>),
}

impl Type {
    pub fn with_view(self) -> Type {
        match self {
            Type::View(ty) => Type::View(ty),
            ty => Type::View(Box::new(ty))
        }
    }

    pub fn is_assignable_from(&self, value: &Type) -> bool {
        match self {
            Type::View(viewn) => value == &**viewn || value == self,
            _ => value == self
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Type::TestType => write!(f, "test"),
            Type::Primitive(PrimitiveType::Int) => write!(f, "int"),
            Type::Array(PrimitiveType::Int, dims) => {
                write!(f, "int[{}]", ",".repeat(*dims as usize))
            }
            Type::Function(params, result) => {
                f.write_str("fn(")?;
                for param in params {
                    param.fmt(f)?;
                    f.write_str(", ")?;
                }
                f.write_str(")")?;
                if let Some(result_type) = result {
                    f.write_str(": ")?;
                    result_type.fmt(f)?;
                }
                Ok(())
            }
            Type::JumpLabel => write!(f, "LABEL"),
            Type::View(viewn_type) => write!(f, "&{}", viewn_type),
        }
    }
}