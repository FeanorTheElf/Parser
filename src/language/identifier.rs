
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Name
{
    name: String,
    id: u32
}

impl Name
{
	pub fn new(name: String) -> Name
	{
        Name {
            name: name,
            id: 0
        }
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BuiltInIdentifier 
{
    FunctionIndex, FunctionAdd, FunctionMul, FunctionUnaryDiv, FunctionUnaryNeg, FunctionAnd, FunctionOr, FunctionLeq, FunctionGeq, FunctionEq, FunctionNeq, FunctionLs, FunctionGt
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Identifier
{
    BuiltIn(BuiltInIdentifier),
    Name(Name)
}
