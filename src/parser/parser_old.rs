use super::super::lexer::tokens::{ Stream, Token };
use super::super::language::prelude::*;
use super::parser_gen::Flatten;
use super::{ Buildable, Build, Parse };

#[cfg(test)]
use super::super::lexer::lexer::lex;

impl_parse!{ Program => Program(Token#BOF { FunctionNode } Token#EOF) }

impl_parse!{ FunctionNode => FunctionNode(Token#Fn identifier Token#BracketOpen { ParameterNode } Token#BracketClose Token#Colon TypeNode FunctionImplementationNode) }
impl_parse!{ dyn FunctionImplementationNode => NativeFunctionNode(Token#Native Token#Semicolon) 
                                             | ImplementedFunctionNode(BlockNode)}
impl_parse!{ ParameterNode => ParameterNode(identifier Token#Colon TypeNode Token#Comma) }
impl_parse!{ BlockNode => BlockNode(Token#CurlyBracketOpen {StmtNode} Token#CurlyBracketClose) }

impl Parse for dyn StmtNode {
	fn guess_can_parse(stream: &Stream) -> bool {
		ExprNode::guess_can_parse(stream) || stream.is_next(&Token::If) || stream.is_next(&Token::While) || stream.is_next(&Token::CurlyBracketOpen) || stream.is_next(&Token::Return) || stream.is_next(&Token::Let)
	}

	fn parse(stream: &mut Stream) -> Result<Box<Self>, CompileError> {
		let pos = stream.pos();
		if IfNode::guess_can_parse(stream) {
			return Ok(Box::new(*IfNode::parse(stream)?));
		} else if WhileNode::guess_can_parse(stream) {
			return Ok(Box::new(*WhileNode::parse(stream)?));
		} else if BlockNode::guess_can_parse(stream) {
			return Ok(BlockNode::parse(stream)?);
		} else if ReturnNode::guess_can_parse(stream) {
			return Ok(Box::new(*ReturnNode::parse(stream)?));
		} else if VariableDeclarationNode::guess_can_parse(stream) {
			return Ok(Box::new(*VariableDeclarationNode::parse(stream)?));
		} else if ExprNode::guess_can_parse(stream) {
			let expr = ExprNode::parse(stream)?;
			if stream.is_next(&Token::Assign) {
				stream.skip_next(&Token::Assign)?;
				let new_val = ExprNode::parse(stream)?;
				stream.skip_next(&Token::Semicolon)?;
				return Ok(Box::new(AssignmentNode::new(pos, expr, new_val)));
			} else {
				stream.skip_next(&Token::Semicolon)?;
				return Ok(Box::new(ExprStmtNode::new(pos, expr)));
			}
		} else {
			return Err(CompileError::new(pos, format!("Expected statement, got {:?} at position {}", stream.peek(), stream.pos()), ErrorType::SyntaxError));
		}
	}
}

impl_parse!{ IfNode => IfNode(Token#If ExprNode BlockNode) }
impl_parse!{ WhileNode => WhileNode(Token#While ExprNode BlockNode) }
impl_parse!{ ReturnNode => ReturnNode(Token#Return ExprNode Token#Semicolon) }
impl_parse!{ VariableDeclarationNode => VariableDeclarationNode(Token#Let identifier Token#Colon TypeNode [ Token#Assign ExprNode ] Token#Semicolon) }

impl Parse for dyn TypeNode {
	fn guess_can_parse(stream: &Stream) -> bool {
		BaseTypeNode::guess_can_parse(stream) || stream.is_next(&Token::Void)
	}

	fn parse(stream: &mut Stream) -> Result<Box<Self>, CompileError>  {
		let pos = stream.pos();
		if stream.is_next(&Token::Void) {
			stream.skip_next(&Token::Void)?;
			return Ok(Box::new(VoidTypeNode::new(pos)));
		} else if BaseTypeNode::guess_can_parse(stream) {
			let base_type = BaseTypeNode::parse(stream)?;
			let mut dimensions: u8 = 0;
			while stream.is_next(&Token::SquareBracketOpen) {
				stream.skip_next(&Token::SquareBracketOpen)?;
				stream.skip_next(&Token::SquareBracketClose)?;
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
		stream.is_next_identifier() || stream.is_next_literal() || stream.is_next(&Token::BracketOpen) || stream.is_next(&Token::New)
	}

	fn parse(stream: &mut Stream) -> Result<Box<Self>, CompileError>  {
		let pos = stream.pos();
		if stream.is_next(&Token::BracketOpen) {
			return Ok(Box::new(*BracketExprNode::parse(stream)?));
		} else if stream.is_next_identifier() {
			let ident = stream.next_ident()?;
			if stream.is_next(&Token::BracketOpen) {
				stream.skip_next(&Token::BracketOpen)?;
				let mut params: AstVec<ExprNode> = vec![];
				while !stream.is_next(&Token::BracketClose) {
					params.push(ExprNode::parse(stream)?);
					stream.skip_next(&Token::Comma)?;
				}
				stream.skip_next(&Token::BracketClose)?;
				return Ok(Box::new(FunctionCallNode::new(pos, ident, params)));
			} else {
				return Ok(Box::new(VariableNode::new(pos, ident)));
			}
		} else if stream.is_next_literal() {
			return Ok(Box::new(LiteralNode::new(pos, stream.next_literal()?)));
		} else if stream.is_next(&Token::New) {
			return Ok(Box::new(*NewExprNode::parse(stream)?));
		} else {
			return Err(CompileError::new(stream.pos(), format!("Expected 'new', '(', identifier or literal, got {:?}", stream.peek()), ErrorType::SyntaxError));
		}
	}
}
impl_parse!{ BracketExprNode => BracketExprNode(Token#BracketOpen ExprNode Token#BracketClose) }
impl_parse!{ NewExprNode => NewExprNode(Token#New BaseTypeNode {IndexPartNode}) }

impl_parse!{ dyn BaseTypeNode => IntTypeNode(Token#Int) }

#[test]
fn test_parse_simple_function() {
    let ident = |name: &'static str| Identifier::new(name);
    let len = *FunctionNode::parse(lex("fn len(a: int[],): int { let b: int[] = a; { return len(b, ); } }").skip_next(&Token::BOF).unwrap()).unwrap();

	assert_eq!(ident("len"), len.ident);
	assert_eq!(1, len.params.len());
	assert_eq!(ident("a"), len.params[0].ident);
	assert_eq!(&ArrTypeNode::test_val(1), &len.params[0].param_type);
	assert_eq!(&ArrTypeNode::test_val(0), &len.result);

	let body = &len.implementation.dynamic().downcast_ref::<ImplementedFunctionNode>().unwrap().body;
	let let_stmt = &body.stmts[0].dynamic().downcast_ref::<VariableDeclarationNode>().unwrap();
	assert_eq!(ident("b"), let_stmt.ident);
	assert_eq!(&ArrTypeNode::test_val(1), &let_stmt.variable_type);

	let return_stmt = &body.stmts[1].dynamic().downcast_ref::<BlockNode>().unwrap().stmts[0].dynamic().downcast_ref::<ReturnNode>().unwrap();
	let function_call = &return_stmt.expr.head.head.head.head.head.head.dynamic().downcast_ref::<FunctionCallNode>().unwrap();
	assert_eq!(ident("len"), function_call.function);
	assert_eq!(1, function_call.params.len());
	
	let param = &function_call.params[0].head.head.head.head.head.head.dynamic().downcast_ref::<VariableNode>().unwrap();
	assert_eq!(ident("b"), param.identifier);
}

#[test]
fn test_parse_ifelse() {
	assert!(FunctionNode::parse(&mut lex("fn len(a: int,): void { if a < 1 {} else if a > 1 {} }")).is_err());
}