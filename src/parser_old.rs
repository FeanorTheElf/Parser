
pub fn parse(stream: &mut Stream) -> Func {
	Func::do_match(stream)
}

trait Parse {
	fn guess_match(stream: &Stream) -> bool;
	fn do_match(stream: &mut Stream) -> Self;
}

impl Parse for ParamDeclaration {
	fn guess_match(stream: &Stream) -> bool {
		stream.ends_ident()
	}

	fn do_match(stream: &mut Stream) -> ParamDeclaration {
		let param_name = stream.next().as_ident();
		stream.expect_next(&Token::Colon);
		let param_type = Type::do_match(stream);
		return ParamDeclaration::ParamDeclaration(param_type, param_name);
	}
}

impl Parse for Func {
	fn guess_match(stream: &Stream) -> bool {
		stream.ends(&Token::Keyword(Keyword::Fn))
	}

	fn do_match(stream: &mut Stream) -> Func {
		stream.expect_next(&Token::Keyword(Keyword::Fn));
		let name = stream.next().as_ident();
		stream.expect_next(&Token::BracketOpen);
		let mut params: Vec<ParamDeclaration> = vec![];
		if ParamDeclaration::guess_match(stream) {
			params.push(ParamDeclaration::do_match(stream));
			while stream.ends(&Token::Comma) {
				stream.expect_next(&Token::Comma);
				params.push(ParamDeclaration::do_match(stream));
			}
		}
		stream.expect_next(&Token::BracketClose);
		let mut return_type = Type::Void;
		if stream.ends(&Token::Colon) {
			stream.expect_next(&Token::Colon);
			return_type = Type::do_match(stream);
		}
		stream.expect_next(&Token::CurlyBracketOpen);
		let stmts = Stmts::do_match(stream);
		stream.expect_next(&Token::CurlyBracketClose);
		return Func::Func(name, params, return_type, stmts);
	}
}

impl Parse for Stmts {
	fn guess_match(stream: &Stream) -> bool {
		Stmt::guess_match(stream)
	}

	fn do_match(stream: &mut Stream) -> Stmts {
		let mut result: Vec<Stmt> = vec![];
		while Stmt::guess_match(stream) {
			result.push(Stmt::do_match(stream));
			stream.expect_next(&Token::Semicolon);
		}
		return Stmts::Stmts(result);
	}
}

impl Parse for Stmt {
	fn guess_match(stream: &Stream) -> bool {
		stream.ends(&Token::Keyword(Keyword::If)) || 
		stream.ends(&Token::Keyword(Keyword::While)) || 
		stream.ends_ident() || 
		stream.ends(&Token::CurlyBracketOpen) ||
		stream.ends(&Token::Keyword(Keyword::Return))
	}

	fn do_match(stream: &mut Stream) -> Stmt {
		if stream.ends(&Token::Keyword(Keyword::If)) {
			stream.expect_next(&Token::Keyword(Keyword::If));
			let condition = Expr::do_match(stream);
			stream.expect_next(&Token::CurlyBracketOpen);
			let stmts = Stmts::do_match(stream);
			stream.expect_next(&Token::CurlyBracketClose);
			return Stmt::If(condition, Box::new(stmts));
		} else if stream.ends(&Token::Keyword(Keyword::While)) {
			stream.expect_next(&Token::Keyword(Keyword::While));
			let condition = Expr::do_match(stream);
			stream.expect_next(&Token::CurlyBracketOpen);
			let stmts = Stmts::do_match(stream);
			stream.expect_next(&Token::CurlyBracketClose);
			return Stmt::If(condition, Box::new(stmts));
		} else if stream.ends(&Token::CurlyBracketOpen) {
			stream.expect_next(&Token::CurlyBracketOpen);
			let stmts = Stmts::do_match(stream);
			stream.expect_next(&Token::CurlyBracketClose);
			return Stmt::Block(Box::new(stmts));
		} else if stream.ends(&Token::Keyword(Keyword::Return)) {
			stream.expect_next(&Token::Keyword(Keyword::Return));
			return Stmt::Return(Expr::do_match(stream));
		} else if Expr::guess_match(stream) {
			let expr = Expr::do_match(stream);
			if stream.ends(&Token::Assign) {
				stream.expect_next(&Token::Assign);
				let newVal = Expr::do_match(stream);
				return Stmt::Assignment(expr, newVal);
			} else if stream.ends(&Token::Colon) {
				if let Expr::Mult(ExprMult::Unary(ExprUn::Basic(BasicExprUn::Variable(name)))) = expr {
					stream.expect_next(&Token::Colon);
					let var_type = Type::do_match(stream);
					if stream.ends(&Token::Assign) {
						stream.expect_next(&Token::Assign);
						let init = Expr::do_match(stream);
						return Stmt::Declaration(var_type, name, Some(init));
					} else {
						return Stmt::Declaration(var_type, name, None);
					}
				} else {
					panic!("Only variables can be declared, got an expression {:?}", expr);
				}
			} else {
				return Stmt::Expr(expr);
			}
		} else {
			panic!("Expected statement, got {:?}", stream);
		}
	}
}

impl Parse for Type {
	fn guess_match(stream: &Stream) -> bool {
		BaseType::guess_match(stream)
	}

	fn do_match(stream: &mut Stream) -> Type {
		let base_type = BaseType::do_match(stream);
		let mut dimension_count: u8 = 0;
		while stream.ends(&Token::SquareBracketOpen) {
			dimension_count += 1;
			stream.expect_next(&Token::SquareBracketOpen);
			stream.expect_next(&Token::SquareBracketClose);
		}
		return Type::Arr(base_type, dimension_count);
	}
}

impl Parse for BaseType {
	fn guess_match(stream: &Stream) -> bool {
		stream.ends(&Token::Keyword(Keyword::Int))
	}

	fn do_match(stream: &mut Stream) -> BaseType {
		stream.expect_next(&Token::Keyword(Keyword::Int));
		return BaseType::Int;
	}
}

impl Parse for Expr {
	fn guess_match(stream: &Stream) -> bool {
		ExprMult::guess_match(stream)
	}

	fn do_match(stream: &mut Stream) -> Expr {
		let first = ExprMult::do_match(stream);
		let mut next: Vec<Summand> = vec![];
		while stream.ends(&Token::OpAdd) || stream.ends(&Token::OpSubtract) {
			next.push(match stream.next() {
				Token::OpAdd => Summand::Pos(ExprMult::do_match(stream)),
				Token::OpSubtract => Summand::Neg(ExprMult::do_match(stream)),
				_ => panic!("Last token of stream should be OpAdd or OpSubtract")
			});
		}
		if next.is_empty() {
			return Expr::Mult(first);
		} else {
			next.insert(0, Summand::Pos(first));
			return Expr::Sum(next);
		}
	}
}

impl Parse for ExprMult {
	fn guess_match(stream: &Stream) -> bool {
		ExprUn::guess_match(stream)
	}

	fn do_match(stream: &mut Stream) -> ExprMult {
		let first = ExprUn::do_match(stream);
		let mut next: Vec<ProductPart> = vec![];
		while stream.ends(&Token::OpMult) || stream.ends(&Token::OpDivide) {
			next.push(match stream.next() {
				Token::OpMult => ProductPart::Factor(ExprUn::do_match(stream)),
				Token::OpDivide => ProductPart::Inverse(ExprUn::do_match(stream)),
				_ => panic!("Last token of stream should be OpAdd or OpSubtract")
			});
		}
		if next.is_empty() {
			return ExprMult::Unary(first);
		} else {
			next.insert(0, ProductPart::Factor(first));
			return ExprMult::Product(next);
		}
	}
}

impl Parse for ExprUn {
	fn guess_match(stream: &Stream) -> bool {
		BasicExprUn::guess_match(stream)
	}

	fn do_match(stream: &mut Stream) -> ExprUn {
		let mut result = ExprUn::Basic(BasicExprUn::do_match(stream));
		while stream.ends(&Token::SquareBracketOpen) {
			stream.expect_next(&Token::SquareBracketOpen);
			result = ExprUn::Indexed(Box::new(result), Box::new(Expr::do_match(stream)));
			stream.expect_next(&Token::SquareBracketClose);
		}
		return result;
	}
}

impl Parse for BasicExprUn {
	fn guess_match(stream: &Stream) -> bool {
		stream.ends(&Token::BracketOpen) || stream.ends_literal() || stream.ends_ident() || stream.ends(&Token::Keyword(Keyword::New))
	}

	fn do_match(stream: &mut Stream) -> BasicExprUn {
		if stream.ends(&Token::BracketOpen) {
			stream.expect_next(&Token::BracketOpen);
			let expr = Expr::do_match(stream);
			stream.expect_next(&Token::BracketClose);
			return BasicExprUn::Brackets(Box::new(expr));
		} else if stream.ends_literal() {
			return BasicExprUn::Literal(stream.next().as_literal());
		} else if stream.ends_ident() {
			let identifier = stream.next().as_ident();
			if stream.ends(&Token::BracketOpen) {
				stream.expect_next(&Token::BracketOpen);
				let mut params: Vec<Expr> = vec![];
				if !stream.ends(&Token::BracketClose) {
					params.push(Expr::do_match(stream));
					while stream.ends(&Token::Comma) {
						stream.expect_next(&Token::Comma);
						params.push(Expr::do_match(stream));
					}
					stream.expect_next(&Token::BracketClose);
				}
				return BasicExprUn::Call(identifier, params);
			} else {
				return BasicExprUn::Variable(identifier);
			}
		} else if stream.ends(&Token::Keyword(Keyword::New)) {
			stream.expect_next(&Token::Keyword(Keyword::New));
			let base_type = BaseType::do_match(stream);
			let mut dimensions: Vec<Expr> = vec![];
			while stream.ends(&Token::SquareBracketOpen) {
				stream.expect_next(&Token::SquareBracketOpen);
				dimensions.push(Expr::do_match(stream));
				stream.expect_next(&Token::SquareBracketClose);
			}
			return BasicExprUn::New(base_type, dimensions);
		} else {
			panic!("Expected expression, got {:?}", stream);
		}
	}
}