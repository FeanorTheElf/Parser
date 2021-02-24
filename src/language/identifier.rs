use std::collections::HashMap;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Name {
    pub name: String,
    pub id: u32,
    pub extra_data: Vec<String>
}

impl Name {
    pub fn new(name: String, id: u32) -> Name {
        Name { 
            name: name, 
            id: id,
            extra_data: Vec::new() 
        }
    }

    #[cfg(test)]
    pub fn l(name: &str) -> Name {
        Name::new(name.to_owned(), 0)
    }

    pub fn components<'a>(&'a self) -> impl 'a + Iterator<Item = String> {
        std::iter::once(&self.name).map(String::clone)
            .chain(std::iter::once(self.id).map(|x| format!("{}", x)))
            .chain(self.extra_data.iter().map(String::clone))
    }
}

impl PartialOrd for Name {
    fn partial_cmp(&self, rhs: &Name) -> Option<std::cmp::Ordering> {
        Some(self.cmp(rhs))
    }
}

impl Ord for Name {
    fn cmp(&self, rhs: &Name) -> std::cmp::Ordering {
        match self.name.cmp(&rhs.name) {
            std::cmp::Ordering::Equal => match self.id.cmp(&rhs.id) {
                std::cmp::Ordering::Equal => self.extra_data.cmp(&rhs.extra_data),
                x => x
            },
            x => x,
        }
    }
}

impl std::fmt::Debug for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}#{}#{:?}", self.name, self.id, self.extra_data)
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.id == 0 {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{}#{}", self.name, self.id)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltInIdentifier {
    FunctionIndex,
    FunctionAdd,
    FunctionMul,
    FunctionUnaryDiv,
    FunctionUnaryNeg,
    FunctionAnd,
    FunctionOr,
    FunctionLeq,
    FunctionGeq,
    FunctionEq,
    FunctionNeq,
    FunctionLs,
    FunctionGt,
    ViewZeros,
}

impl BuiltInIdentifier {
    pub fn get_symbol(&self) -> &'static str {

        match self {
            BuiltInIdentifier::FunctionIndex => "[]",
            BuiltInIdentifier::FunctionAdd => "+",
            BuiltInIdentifier::FunctionMul => "*",
            BuiltInIdentifier::FunctionUnaryDiv => "/",
            BuiltInIdentifier::FunctionUnaryNeg => "-",
            BuiltInIdentifier::FunctionAnd => "&&",
            BuiltInIdentifier::FunctionOr => "||",
            BuiltInIdentifier::FunctionLeq => "<=",
            BuiltInIdentifier::FunctionGeq => ">=",
            BuiltInIdentifier::FunctionEq => "==",
            BuiltInIdentifier::FunctionNeq => "!=",
            BuiltInIdentifier::FunctionLs => "<",
            BuiltInIdentifier::FunctionGt => ">",
            BuiltInIdentifier::ViewZeros => "zeros"
        }
    }

    pub fn is_unary_function(&self) -> bool {

        *self == BuiltInIdentifier::FunctionUnaryDiv || *self == BuiltInIdentifier::FunctionUnaryNeg
    }
}

impl std::fmt::Display for BuiltInIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.get_symbol())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Identifier {
    BuiltIn(BuiltInIdentifier),
    Name(Name),
}

impl std::fmt::Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Identifier::BuiltIn(op) => write!(f, "operator {}", op),
            Identifier::Name(name) => write!(f, "{}", name)
        }
    }
}

impl Identifier {
    pub fn unwrap_name(&self) -> &Name {
        match self {
            Identifier::BuiltIn(op) => panic!("Called unwrap_name() on builtin identifier {}", op),
            Identifier::Name(name) => name,
        }
    }

    pub fn unwrap_builtin(&self) -> &BuiltInIdentifier {
        match self {
            Identifier::BuiltIn(op) => op,
            Identifier::Name(name) => panic!("Called unwrap_builtin() on name {}", name),
        }
    }

    pub fn is_name(&self) -> bool {
        match self {
            Identifier::BuiltIn(_) => false,
            Identifier::Name(_) => true
        }
    }

    pub fn is_builtin(&self) -> bool {
        !self.is_name()
    }

    pub fn create(name: Name) -> Self {
        let named_builtin_identifiers = singleton!{
            {{
                let mut result = HashMap::new();
                result.insert(Name::new("zeros".to_owned(), 0), BuiltInIdentifier::ViewZeros);
                result
            }}: HashMap<Name, BuiltInIdentifier>
        };
        if let Some(builtin_identifier) = named_builtin_identifiers.get(&name) {
            Identifier::BuiltIn(*builtin_identifier)
        } else {
            Identifier::Name(name)
        }
    }
}
