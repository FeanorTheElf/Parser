use super::super::language::prelude::*;

impl CompileError {

    pub fn expected_nonvoid(pos: &TextPosition) -> CompileError {
        CompileError::new(
            pos, 
            format!("Value expected, but got void"),
            ErrorType::TypeError
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TypeNotYetCalculated;

pub trait TypeStored {
    fn get_stored_type_if_calculated(&self) -> Result<Option<&Type>, TypeNotYetCalculated>;

    fn get_stored_type(&self) -> &Type {
        self.get_stored_type_if_calculated().unwrap().unwrap()
    }

    fn get_voidable_stored_type(&self) -> Option<&Type> {
        self.get_stored_type_if_calculated().unwrap()
    }
}

impl TypeStored for FunctionCall {
    fn get_stored_type_if_calculated(&self) -> Result<Option<&Type>, TypeNotYetCalculated> {
        self.result_type_cache.as_ref().map(Option::as_ref).ok_or(TypeNotYetCalculated)
    }
}

impl TypeStored for Literal {
    fn get_stored_type_if_calculated(&self) -> Result<Option<&Type>, TypeNotYetCalculated> {
        Ok(Some(&self.literal_type))
    }
}

pub fn get_expression_type_if_calculated<'a, 'b, D: DefinitionEnvironment<'a, 'a>>(expr: &'b Expression, scopes: &'b D) -> Result<Option<&'b Type>, TypeNotYetCalculated> {
    match expr {
        Expression::Call(call) => {
            call.get_stored_type_if_calculated()
        },
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => Ok(Some(scopes.get_defined(name, var.pos()).internal_error().get_type())),
            Identifier::BuiltIn(op) => panic!("Called calculate_type() on builtin identifier {:?}, but builtin identifiers can have different types depending on context", op)
        },
        Expression::Literal(lit) => lit.get_stored_type_if_calculated()
    }
}

pub fn get_expression_type<'a, 'b, D: DefinitionEnvironment<'a, 'a>>(expr: &'b Expression, scopes: &'b D) -> Option<&'b Type> {
    get_expression_type_if_calculated(expr, scopes).unwrap()
}

pub fn get_expression_type_nonvoid<'a, 'b, D: DefinitionEnvironment<'a, 'a>>(expr: &'b Expression, scopes: &'b D) -> Result<&'b Type, CompileError> {
    get_expression_type(expr, scopes).ok_or_else(|| CompileError::expected_nonvoid(expr.pos()))
}