use super::ast::*;
use super::super::lexer::tokens::*;
use super::super::lexer::error::CompileError;
#[macro_use]
use super::parser_gen::{ Parse, Flatten };

use std::vec::Vec;

impl_parse!{ FunctionNode => FunctionNode(Token#Fn identifier Token#BracketOpen {ParameterNode} Token#BracketClose Token#Colon TypeNode Token#CurlyBracketOpen StmtsNode Token#CurlyBracketClose) }
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
			panic!("Expected type, got {:?} at position {}", stream.peek(), stream.pos());
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
			panic!("Expected 'new', '(', identifier or literal, got {:?} at position {}", stream.peek(), stream.pos());
		}
	}
}
impl_parse!{ BracketExprNode => BracketExprNode(Token#BracketOpen ExprNode Token#BracketClose) }
impl_parse!{ NewExprNode => NewExprNode(Token#New BaseTypeNode {IndexPartNode}) }

impl_parse!{ dyn BaseTypeNode => IntTypeNode(Token#Int) }
