use super::ast::{Func, Stmts, Stmt, Type, Expr, Summand, ExprMult, ProductPart, ExprUn, BasicExprUn, ParamDeclaration, BaseType};
use super::tokens::{Identifier, Literal, Keyword, Token, Stream};

use std::string::String;
use std::collections::HashMap;

trait Generate {
	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String;
}

pub fn as_rust_code(ast: Func) -> String {
	Func::generate(ast, &mut HashMap::new())
}

impl Generate for Func {
	
	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			Func::Func(name, mut params, return_type, stmts) => {
				let mut param_names: Vec<String> = vec![];
				for param in &params {
					match param {
						ParamDeclaration::ParamDeclaration(param_type, ident) => {
							param_names.push(ident.name.clone());
							is_var_ref.insert(ident.name.clone(), true);
						}
					};
				}
				is_var_ref.insert(name.name.clone(), true);
				let result = format!("fn {}({}) -> {} {}\n{}{}", name.name, 
					params.drain(..).map(|param|ParamDeclaration::generate(param, is_var_ref)).collect::<Vec<String>>().join(", "), 
					Type::generate(return_type, is_var_ref), "{", Stmts::generate(stmts, is_var_ref), "}");	
				for name in param_names {
					is_var_ref.remove(&name);
				}
				result
			}
		}
	}
}

impl Generate for Type {

	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			Type::Arr(base_type, dim) => {
				let mut result = BaseType::generate(base_type, is_var_ref);
				for i in 0..dim {
					result = format!("Vec<{}>", result);
				}
				result
			},
			Type::Void => "()".to_string()
		}
	}
}

impl Generate for BaseType {

	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			BaseType::Int => "i32".to_string()
		}
	}
}

impl Generate for ParamDeclaration {

	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			ParamDeclaration::ParamDeclaration(param_type, name) => format!("{}: &mut {}", name.name, Type::generate(param_type, is_var_ref))
		}
	}
}

impl Generate for Stmts {

	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			Stmts::Stmts(mut data) => {
				let mut local_vars: Vec<String> = vec![];
				let result = data.drain(..).map(|stmt|generate_stmt(stmt, is_var_ref, &mut local_vars)).collect::<Vec<String>>().join(";\n") + ";\n";
				for name in local_vars {
					is_var_ref.remove(&name);
				};
				return result;
			}
		}
	}
}

fn generate_stmt(ast: Stmt, is_var_ref: &mut HashMap<String, bool>, local_vars: &mut Vec<String>) -> String {
	match ast {
		Stmt::Declaration(vtype, name, expr) => {
			is_var_ref.insert(name.name.clone(), false);
			local_vars.push(name.name.clone());
			let result = format!("let mut {}: {} = {}", name.name, 
				Type::generate(vtype, is_var_ref), Expr::generate(expr.unwrap(), is_var_ref));
			result
		},
		Stmt::Assignment(name, expr) => 
			format!("{} = {}", Expr::generate(name, is_var_ref), Expr::generate(expr, is_var_ref)),
		Stmt::Expr(expr) => 
			Expr::generate(expr, is_var_ref),
		Stmt::If(expr, body) => 
			format!("if {} {}\n{} {}", Expr::generate(expr, is_var_ref), "{", Stmts::generate(*body, is_var_ref), "}"),
		Stmt::While(expr, body) => 
			format!("while {} {}\n{} {}", Expr::generate(expr, is_var_ref), "{", Stmts::generate(*body, is_var_ref), "}"),
		Stmt::Block(body) => 
			format!("{}\n{} {}", "{", Stmts::generate(*body, is_var_ref), "}"),
		Stmt::Return(expr) =>
			format!("return {}", Expr::generate(expr, is_var_ref))
	}
}

impl Generate for Expr {

	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			Expr::Mult(expr) => ExprMult::generate(expr, is_var_ref),
			Expr::Sum(mut parts) => parts.drain(..).map(|part|Summand::generate(part, is_var_ref)).collect::<Vec<String>>().join(" + ")
		}
	}
}

impl Generate for Summand {

	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			Summand::Pos(expr) => format!("{}", ExprMult::generate(expr, is_var_ref)),
			Summand::Neg(expr) => format!("-({})", ExprMult::generate(expr, is_var_ref))
		}
	}
}

impl Generate for ExprMult {

	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			ExprMult::Unary(expr) => ExprUn::generate(expr, is_var_ref),
			ExprMult::Product(mut parts) => parts.drain(..).map(|part|ProductPart::generate(part, is_var_ref)).collect::<Vec<String>>().join(" * ")
		}
	}
}

impl Generate for ProductPart {

	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			ProductPart::Factor(expr) => format!("{}", ExprUn::generate(expr, is_var_ref)),
			ProductPart::Inverse(expr) => format!("1/({})", ExprUn::generate(expr, is_var_ref))
		}
	}
}

impl Generate for ExprUn {

	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			ExprUn::Basic(expr) => BasicExprUn::generate(expr, is_var_ref),
			ExprUn::Indexed(main, index) => format!("({})[{}]", ExprUn::generate(*main, is_var_ref), Expr::generate(*index, is_var_ref))
		}
	}
}

impl Generate for BasicExprUn {

	fn generate(ast: Self, is_var_ref: &mut HashMap<String, bool>) -> String {
		match ast {
			BasicExprUn::Brackets(expr) => format!("({})", Expr::generate(*expr, is_var_ref)),
			BasicExprUn::Literal(lit) => lit.value.to_string(),
			BasicExprUn::Variable(ident) => {
				if let Some(is_ref) = is_var_ref.get(&ident.name) {
					if *is_ref {
						format!("*{}", ident.name)
					} else {
						ident.name
					}
				} else {
					panic!("Unknown variable: {}", ident.name)
				}
			},
			BasicExprUn::New(base_type, mut dimensions) => {
				let mut length_var_defs = String::new();
				let mut current_type = BaseType::generate(base_type, is_var_ref);
				let mut result = format!("0");
				let mut counter = dimensions.len();
				while let Some(dim_expr) = dimensions.pop() {
					length_var_defs += &format!("let __create_vec_len_{}: usize = ({}) as usize;", counter, Expr::generate(dim_expr, is_var_ref));
					current_type = format!("Vec<{}>", current_type);
					result = format!("{} let mut vec: {} = Vec::new(); vec.resize_with(__create_vec_len_{}, ||{}); vec {}", "{", current_type, counter, result, "}");
					counter -= 1;
				}
				format!("{} {} {} {}", "{", length_var_defs, result, "}")
			},
			BasicExprUn::Call(func, mut params) => format!("{}({})", func.name, 
				params.drain(..).map(|expr|Expr::generate(expr, is_var_ref)).collect::<Vec<String>>().join(", "))
		}
	}
}