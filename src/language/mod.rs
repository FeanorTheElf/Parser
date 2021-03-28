pub mod position;

pub mod identifier;
pub mod error;
pub mod scopes;
pub mod ast;
pub mod types;
pub mod ast_expr;
pub mod ast_statement;
pub mod ast_ifwhile;
pub mod ast_return;
pub mod ast_assignment;
pub mod ast_function;
pub mod ast_pfor;
pub mod prelude;
pub mod compiler;
pub mod gwaihir_writer;

#[cfg(test)]
pub mod ast_test;