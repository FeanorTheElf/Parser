use super::super::language::prelude::*;
use super::super::language::concrete_views::*;
use super::super::language::gwaihir_writer::*;
use super::super::language::ast_assignment::*;
use super::super::language::ast_pfor::*;
use super::super::language::ast_return::*;
use super::types::*;

impl CompileError {

    fn not_initializable(pos: &TextPosition, variable: &Type, value: &Type) -> CompileError {
        CompileError::new(
            pos,
            format!("Cannot initialize a variable of type {} with a value of type {}", DisplayWrapper::from(variable), DisplayWrapper::from(value)),
            ErrorType::TypeError
        )
    }

    fn not_returnable(pos: &TextPosition, return_type: Option<&Type>, value: Option<&Type>) -> CompileError {
        match (value, return_type) {
            (Some(v), Some(r)) => CompileError::new(
                pos,
                format!("Cannot return a value of type {} from a function with return type {}", DisplayWrapper::from(v), DisplayWrapper::from(r)),
                ErrorType::TypeError
            ),
            (None, Some(r)) => CompileError::new(
                pos,
                format!("This function must return a value of type {}", DisplayWrapper::from(r)),
                ErrorType::TypeError
            ),
            (Some(_v), None) => CompileError::new(
                pos,
                format!("Cannot return a value from a void function"),
                ErrorType::TypeError
            ),
            (None, None) => panic!("returning void from a void function is always legal")
        }
    }

    fn not_assignable(pos: &TextPosition, variable: &Type, value: &Type) -> CompileError {
        CompileError::new(
            pos,
            format!("Cannot assign a value of type {} to an expression of type {}", DisplayWrapper::from(value), DisplayWrapper::from(variable)),
            ErrorType::TypeError
        )
    }
}

pub trait TypecheckedStatement {

    ///
    /// Checks that the statement is valid with the current expressions and initializes
    /// any type-dependent data in the statement.
    /// 
    /// # Details
    /// 
    /// This function is called during typechecking, when the types in all expressions occuring
    /// in the statement have been determined and typechecked. Therefore, this function only has
    /// to check the statement-specific semantic of this statement (e.g. that the value of an 
    /// assignment is assignable to the corresponding variable, but not that any function calls
    /// that give the value have valid parameter). It may use `get_expression_type()` to get
    /// the type of expressions in the statement (however types of other expression might not
    /// yet have been stored).
    ///  
    fn determine_types<'a, D>(&mut self, scopes: &D) -> CompileError
        where D: DefinitionEnvironment<'a, 'a>;
}

pub fn typecheck_statement<'a, D>(statement: &mut dyn Statement, scopes: &D, parent_function_type: &FunctionType) -> Result<(), CompileError>
    where D: DefinitionEnvironment<'a, 'a>
{
    if let Some(decl) = statement.downcast_mut::<LocalVariableDeclaration>() {
        let assigned_type = get_expression_type_nonvoid(&decl.value, scopes)?;
        let target_type = decl.declaration.get_type_mut();
        if !target_type.is_initializable_by(target_type) {
            Err(CompileError::not_initializable(&decl.value.pos(), target_type, assigned_type))?;
        }
        if target_type.is_view() && assigned_type.is_view() {
            target_type.as_view_mut().unwrap().concrete_view = Some(assigned_type.as_view().unwrap().get_concrete().dyn_clone());
        } else if target_type.is_view() && !assigned_type.is_view() {
            target_type.as_view_mut().unwrap().concrete_view = Some(Box::new(VIEW_REFERENCE));
        }
    } else if let Some(assignment) = statement.downcast_mut::<Assignment>() {
        let assigned_type = get_expression_type_nonvoid(&assignment.value, scopes)?;
        let assignee_type = get_expression_type_nonvoid(&assignment.assignee, scopes)?;
        if !assigned_type.is_copyable_to(assignee_type) {
            Err(CompileError::not_assignable(&assignment.pos(), assignee_type, assigned_type))?;
        }
    } else if let Some(ret) = statement.downcast_mut::<Return>() {
        let returned_type = ret.value.as_ref().map(|e| get_expression_type_nonvoid(e, scopes)).transpose()?;
        if returned_type.is_some() && parent_function_type.return_type().is_some() {
            if !parent_function_type.return_type().unwrap().is_initializable_by(returned_type.unwrap()) {
                Err(CompileError::not_returnable(&ret.pos(), parent_function_type.return_type(), returned_type))?;
            }
        } else if returned_type.is_some() != parent_function_type.return_type().is_some() {
            Err(CompileError::not_returnable(&ret.pos(), parent_function_type.return_type(), returned_type))?;
        }
    } else if let Some(pfor) = statement.downcast_mut::<ParallelFor>() {
        for array_access in pfor.access_pattern_mut()? {
            array_access.store_array_entry_types(scopes)?;
        }
    }
    return Ok(());
}