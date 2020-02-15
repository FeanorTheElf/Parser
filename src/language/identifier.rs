
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Name
{
    pub name: String,
    pub id: u32
}

impl Name
{
	pub fn new(name: String, id: u32) -> Name
	{
        Name {
            name: name,
            id: id
        }
    }

    #[cfg(test)]
    pub fn l(name: &'static str) -> Name
    {
        Name::new(name.to_owned(), 0)
    }
}

impl PartialOrd for Name
{
	fn partial_cmp(&self, rhs: &Name) -> Option<std::cmp::Ordering>
	{
		Some(self.cmp(rhs))
	}
}

impl Ord for Name
{
	fn cmp(&self, rhs: &Name) -> std::cmp::Ordering
	{
        match self.name.cmp(&rhs.name) {
            std::cmp::Ordering::Equal => self.id.cmp(&rhs.id),
            x => x
        }
	}
}

impl std::fmt::Display for Name
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
    {
        write!(f, "{}#{}", self.name, self.id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltInIdentifier 
{
    FunctionIndex, FunctionAdd, FunctionMul, FunctionUnaryDiv, FunctionUnaryNeg, FunctionAnd, FunctionOr, FunctionLeq, FunctionGeq, FunctionEq, FunctionNeq, FunctionLs, FunctionGt
}

impl std::fmt::Display for BuiltInIdentifier
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
    {
        write!(f, "{}", match self {
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
            BuiltInIdentifier::FunctionGt => ">"
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Identifier
{
    BuiltIn(BuiltInIdentifier),
    Name(Name)
}
