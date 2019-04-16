use super::ast::*;
use super::tokens::*;

use std::string::String;
use std::collections::HashMap;

trait Generate {
	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String;
}
