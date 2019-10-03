use super::prelude::*;
use super::super::lexer::tokens::{ Stream, Token };
use super::Parse;
#[macro_use]
use super::parser_gen::Flatten;

#[cfg(test)]
use super::super::lexer::lexer::lex;

use std::vec::Vec;

impl_parse!{ FunctionNode => FunctionNode(Token#Fn identifier Token#BracketOpen {ParameterNode} Token#BracketClose Token#Colon TypeNode FunctionImplementationNode) }
impl_parse!{ dyn FunctionImplementationNode => NativeFunctionNode(Token#Native Token#Semicolon) 
                                             | ImplementedFunctionNode(Token#CurlyBracketOpen StmtsNode Token#CurlyBracketClose)}
impl_parse!{ ParameterNode => ParameterNode(identifier Token#Colon TypeNode Token#Comma) }
impl_parse!{ StmtsNode => StmtsNode({StmtNode}) }

impl Parse for dyn StmtNode {
	fn guess_can_parse(stream: &Stream) -> bool {
		ExprNode::guess_can_parse(stream) || stream.ends(&Token::If) || stream.ends(&Token::While) || stream.ends(&Token::CurlyBracketOpen) || stream.ends(&Token::Return) || stream.ends(&Token::Let)
	}

	fn parse(stream: &mut Stream) -> Result<Box<Self>, CompileError> {
		let pos = stream.pos();
		if stream.ends(&Token::If) {
			return Ok(Box::new(*IfNode::parse(stream)?));
		} else if stream.ends(&Token::While) {
			return Ok(Box::new(*WhileNode::parse(stream)?));
		} else if stream.ends(&Token::CurlyBracketOpen) {
			return Ok(Box::new(*BlockNode::parse(stream)?));
		} else if stream.ends(&Token::Return) {
			return Ok(Box::new(*ReturnNode::parse(stream)?));
		} else if stream.ends(&Token::Let) {
			return Ok(Box::new(*VariableDeclarationNode::parse(stream)?));
		} else if ExprNode::guess_can_parse(stream) {
			let expr = ExprNode::parse(stream)?;
			if stream.ends(&Token::Assign) {
				stream.expect_next(&Token::Assign);
				let new_val = ExprNode::parse(stream)?;
				stream.expect_next(&Token::Semicolon);
				return Ok(Box::new(AssignmentNode::new(pos, expr, new_val)));
			} else {
				stream.expect_next(&Token::Semicolon);
				return Ok(Box::new(ExprStmtNode::new(pos, expr)));
			}
		} else {
			panic!("Expected statement, got {:?} at position {}", stream.peek(), stream.pos());
		}
	}
}

impl_parse!{ IfNode => IfNode(Token#If ExprNode Token#CurlyBracketOpen StmtsNode Token#CurlyBracketClose) }
impl_parse!{ WhileNode => WhileNode(Token#If ExprNode Token#CurlyBracketOpen StmtsNode Token#CurlyBracketClose) }
impl_parse!{ BlockNode => BlockNode(Token#CurlyBracketOpen StmtsNode Token#CurlyBracketClose) }
impl_parse!{ ReturnNode => ReturnNode(Token#Return ExprNode Token#Semicolon) }
impl_parse!{ VariableDeclarationNode => VariableDeclarationNode(Token#Let identifier Token#Colon TypeNode Token#Assign ExprNode Token#Semicolon) }

impl Parse for dyn TypeNode {
	fn guess_can_parse(stream: &Stream) -> bool {
		BaseTypeNode::guess_can_parse(stream) || stream.ends(&Token::Void)
	}

	fn parse(stream: &mut Stream) -> Result<Box<Self>, CompileError>  {
		let pos = stream.pos();
		if stream.ends(&Token::Void) {
			stream.expect_next(&Token::Void)?;
			return Ok(Box::new(VoidTypeNode::new(pos)));
		} else if BaseTypeNode::guess_can_parse(stream) {
			let base_type = BaseTypeNode::parse(stream)?;
			let mut dimensions: u8 = 0;
			while stream.ends(&Token::SquareBracketOpen) {
				stream.expect_next(&Token::SquareBracketOpen);
				stream.expect_next(&Token::SquareBracketClose);
				dimensions += 1;
			}
			return Ok(Box::new(ArrTypeNode::new(pos, base_type, dimensions)));
		} else {
			return Err(CompileError::new(stream.pos(), format!("Expected type, got {:?}", stream.peek()), ErrorType::SyntaxError));
		}
	}
}

impl_parse!{ ExprNodeLvlOr => ExprNodeLvlOr(ExprNodeLvlAnd {OrPartNode}) }
impl_parse!{ OrPartNode => OrPartNode(Token#OpOr ExprNodeLvlAnd) }

impl_parse!{ ExprNodeLvlAnd => ExprNodeLvlAnd(ExprNodeLvlCmp {AndPartNode}) }
impl_parse!{ AndPartNode => AndPartNode(Token#OpAnd ExprNodeLvlCmp) }

impl_parse!{ ExprNodeLvlCmp => ExprNodeLvlCmp(ExprNodeLvlAdd {CmpPartNode}) }
impl_parse!{ dyn CmpPartNode => CmpPartNodeEq(Token#OpEqual ExprNodeLvlAdd)
                              | CmpPartNodeNeq(Token#OpUnequal ExprNodeLvlAdd)
                              | CmpPartNodeLeq(Token#OpLessEq ExprNodeLvlAdd)
                              | CmpPartNodeGeq(Token#OpGreaterEq ExprNodeLvlAdd)
                              | CmpPartNodeLs(Token#OpLess ExprNodeLvlAdd)
					          | CmpPartNodeGt(Token#OpGreater ExprNodeLvlAdd) }

impl_parse!{ ExprNodeLvlAdd => ExprNodeLvlAdd(ExprNodeLvlMult {SumPartNode}) }
impl_parse!{ dyn SumPartNode => SumPartNodeAdd(Token#OpAdd ExprNodeLvlMult)
                              | SumPartNodeSub(Token#OpSubtract ExprNodeLvlMult) }

impl_parse!{ ExprNodeLvlMult => ExprNodeLvlMult(ExprNodeLvlIndex {ProductPartNode}) }
impl_parse!{ dyn ProductPartNode => ProductPartNodeMult(Token#OpMult ExprNodeLvlIndex)
                                  | ProductPartNodeDivide(Token#OpDivide ExprNodeLvlIndex) }

impl_parse!{ ExprNodeLvlIndex => ExprNodeLvlIndex(UnaryExprNode {IndexPartNode}) }
impl_parse!{ IndexPartNode => IndexPartNode(Token#SquareBracketOpen ExprNode Token#SquareBracketClose) }

impl Parse for dyn UnaryExprNode {
	fn guess_can_parse(stream: &Stream) -> bool {
		stream.ends_ident() || stream.ends_literal() || stream.ends(&Token::BracketOpen) || stream.ends(&Token::New)
	}

	fn parse(stream: &mut Stream) -> Result<Box<Self>, CompileError>  {
		let pos = stream.pos();
		if stream.ends(&Token::BracketOpen) {
			return Ok(Box::new(*BracketExprNode::parse(stream)?));
		} else if stream.ends_ident() {
			let ident = stream.next_ident()?;
			if stream.ends(&Token::BracketOpen) {
				stream.expect_next(&Token::BracketOpen);
				let mut params: AstVec<ExprNode> = vec![];
				if !stream.ends(&Token::BracketClose) {
					params.push(ExprNode::parse(stream)?);
					while stream.ends(&Token::Comma) {
						stream.expect_next(&Token::Comma);
						params.push(ExprNode::parse(stream)?);
					}
				}
				stream.expect_next(&Token::BracketClose);
				return Ok(Box::new(FunctionCallNode::new(pos, ident, params)));
			} else {
				return Ok(Box::new(VariableNode::new(pos, ident)));
			}
		} else if stream.ends_literal() {
			return Ok(Box::new(LiteralNode::new(pos, stream.next_literal()?)));
		} else if stream.ends(&Token::New) {
			return Ok(Box::new(*NewExprNode::parse(stream)?));
		} else {
			return Err(CompileError::new(stream.pos(), format!("Expected 'new', '(', identifier or literal, got {:?}", stream.peek()), ErrorType::SyntaxError));
		}
	}
}
impl_parse!{ BracketExprNode => BracketExprNode(Token#BracketOpen ExprNode Token#BracketClose) }
impl_parse!{ NewExprNode => NewExprNode(Token#New BaseTypeNode {IndexPartNode}) }

impl_parse!{ dyn BaseTypeNode => IntTypeNode(Token#Int) }

#[cfg(test)]
fn create_int_arr(dims: u8) -> Box<dyn TypeNode> {
	Box::new(ArrTypeNode::new(TextPosition::create(0, 0), Box::new(IntTypeNode::new(TextPosition::create(0, 0))), dims))
}

#[test]
fn test_parse_simple_function() {
	let default_pos = TextPosition::create(0, 0);
    let ident = |name: &'static str| Identifier { name: name.to_owned() };
    let len = *FunctionNode::parse(&mut lex("fn len(a: int[],): int { let b: int[] = a; { return len(b); } }".to_owned())).unwrap();

	assert_eq!(ident("len"), len.ident);
	assert_eq!(1, len.params.len());
	assert_eq!(ident("a"), len.params[0].ident);
	assert_eq!(&create_int_arr(1), &len.params[0].param_type);
	assert_eq!(&create_int_arr(0), &len.result);

	let body = &len.implementation.dynamic().downcast_ref::<ImplementedFunctionNode>().unwrap().stmts;
	let let_stmt = &body.stmts[0].dynamic().downcast_ref::<VariableDeclarationNode>().unwrap();
	assert_eq!(ident("b"), let_stmt.ident);
	assert_eq!(&create_int_arr(1), &let_stmt.variable_type);

	let return_stmt = &body.stmts[1].dynamic().downcast_ref::<BlockNode>().unwrap().block.stmts[0].dynamic().downcast_ref::<ReturnNode>().unwrap();
	let function_call = &return_stmt.expr.head.head.head.head.head.head.dynamic().downcast_ref::<FunctionCallNode>().unwrap();
	assert_eq!(ident("len"), function_call.function);
	assert_eq!(1, function_call.params.len());
	
	let param = &function_call.params[0].head.head.head.head.head.head.dynamic().downcast_ref::<VariableNode>().unwrap();
	assert_eq!(ident("b"), param.identifier);
}