use super::super::language::prelude::*;
use super::super::language::ast_ifwhile::*;
use super::super::language::ast_return::*;
use super::super::language::ast_assignment::*;
use super::super::language::ast_pfor::*;
use super::super::language::ast_goto::*;
use super::super::lexer::tokens::{Stream, Token};
use super::parser_gen::*;
use super::{Build, Parseable, Parser, TopLevelParser, ParserContext};

impl Parseable for Program {
    type ParseOutputType = Self;
}

impl TopLevelParser for Program {
    
    fn parse(stream: &mut Stream) -> Result<Program, CompileError> {
        let mut context = ParserContext::new();
        stream.skip_next(&Token::BOF)?;
        let mut result = Program {
            items: Vec::new()
        };
        while Function::is_applicable(stream) {
            result.items.push(Function::parse(stream, &mut context)?);
        }
        stream.skip_next(&Token::EOF)?;
        return Ok(result);
    }
}

impl Parseable for Type {
    type ParseOutputType = Type;
}

impl Build<TypeNodeNoView> for Type {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: TypeNodeNoView) -> Self::ParseOutputType {
        let mutable = match (param.1).0 {
            Some(ReadWrite::TypeWrite(_)) => true,
            _ => false
        };
        let base = match (param.1).1 {
            PrimitiveTypeNode::IntTypeNode(_) => PrimitiveType::Int,
            PrimitiveTypeNode::FloatTypeNode(_) => PrimitiveType::Float
        };
        let dims = (param.1).2.map(|x| (x.1).0 as usize).unwrap_or(0);
        return Type::array_type(base, dims, mutable);
    }
}

impl Build<TypeNodeView> for Type {
    fn build(pos: TextPosition, context: &mut ParserContext, param: TypeNodeView) -> Self::ParseOutputType {
        let type_node = (param.1).0;
        let viewn_type = Type::build(pos, context, type_node);
        return Type::with_view(viewn_type);
    }
}

impl Parseable for Name {
    type ParseOutputType = Self;
}

impl Parser for Name {
    fn is_applicable(stream: &Stream) -> bool {

        stream.is_next_identifier()
    }

    fn parse(stream: &mut Stream, _context: &mut ParserContext) -> Result<Self::ParseOutputType, CompileError> {

        let identifier_string = stream.next_ident()?;

        if let Some(index) = identifier_string.find('#') {

            let id = identifier_string[index+1..].parse::<u32>().map_err(|_| CompileError::new(stream.pos(), format!("
                Invalid identifier: Must be an alphanumeric sequence (and `_`), optionally followed by an index of the form #<number>, got {}", 
                identifier_string), ErrorType::IncorrectIdentifier))?;

            Ok(Name::new(identifier_string[0..index].to_owned(), id))
        } else {

            Ok(Name::new(identifier_string, 0))
        }
    }
}

impl Parseable for Variable {
    type ParseOutputType = Self;
}

impl Build<Name> for Variable {
    fn build(pos: TextPosition, _context: &mut ParserContext, param: Name) -> Self::ParseOutputType {

        Variable {
            pos: pos,
            identifier: Identifier::create(param),
        }
    }
}

impl Parseable for Literal {
    type ParseOutputType = Self;
}

impl Parser for Literal {
    fn is_applicable(stream: &Stream) -> bool {

        stream.is_next_literal()
    }

    fn parse(stream: &mut Stream, _context: &mut ParserContext) -> Result<Self::ParseOutputType, CompileError> {
        Ok(Literal {
            pos: stream.pos().clone(),
            value: stream.next_literal()?,
            literal_type: SCALAR_INT
        })
    }
}

impl Parseable for Declaration {
    type ParseOutputType = Self;
}

impl Build<(Name, <Type as Parseable>::ParseOutputType)> for Declaration {
    fn build(pos: TextPosition, _context: &mut ParserContext, param: (Name, <Type as Parseable>::ParseOutputType)) -> Self::ParseOutputType {
        Declaration {
            pos: pos,
            var_type: param.1,
            name: param.0,
        }
    }
}

impl Parseable for Function {
    type ParseOutputType = Self;
}

impl Build<(Name, Vec<DeclarationListNode>, Option<<Type as Parseable>::ParseOutputType>, FunctionImpl)> for Function {
    fn build(
        pos: TextPosition, context: &mut ParserContext,
        param: (Name, Vec<DeclarationListNode>, Option<<Type as Parseable>::ParseOutputType>, FunctionImpl),
    ) -> Self::ParseOutputType {

        let block = if let FunctionImpl::Block(block) = param.3 {
            Some(block)
        } else {
            None
        };
        let params: Vec<Declaration> = param.1.into_iter().map(|p| Declaration::build(p.0, context, p.1)).collect();
        let return_type = param.2;

        Function::new(pos, param.0, params, return_type, block)
    }
}

impl Parseable for Block {
    type ParseOutputType = Self;
}

impl Build<(Vec<Box<dyn Statement>>,)> for Block {
    fn build(pos: TextPosition, _context: &mut ParserContext, param: (Vec<Box<dyn Statement>>,)) -> Self::ParseOutputType {
        Block::new(pos, param.0)
    }
}

impl Parseable for dyn Statement {
    type ParseOutputType = Box<Self>;
}

impl Build<ExpressionNode> for dyn Statement {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: ExpressionNode) -> Self::ParseOutputType {
        Box::new((param.1).0)
    }
}

impl Parseable for If {
    type ParseOutputType = Self;
}

impl Build<(<Expression as Parseable>::ParseOutputType, Block)> for If {
    fn build(
        pos: TextPosition, _context: &mut ParserContext,
        param: (<Expression as Parseable>::ParseOutputType, Block),
    ) -> Self::ParseOutputType {
        If::new(pos, param.0, param.1)
    }
}

impl Build<If> for dyn Statement {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: If) -> Self::ParseOutputType {
        Box::new(param)
    }
}

impl Parseable for While {
    type ParseOutputType = Self;
}

impl Build<(<Expression as Parseable>::ParseOutputType, Block)> for While {
    fn build(
        pos: TextPosition, _context: &mut ParserContext,
        param: (<Expression as Parseable>::ParseOutputType, Block),
    ) -> Self::ParseOutputType {
        While::new(pos, param.0, param.1)
    }
}

impl Build<While> for dyn Statement {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: While) -> Self::ParseOutputType {
        Box::new(param)
    }
}

// impl Parseable for Label {
//     type ParseOutputType = Self;
// }

// impl Build<(Name,)> for Label {
//     fn build(pos: TextPosition, _context: &mut ParserContext, param: (Name,)) -> Self::ParseOutputType {

//         Label {
//             pos: pos,
//             label: param.0,
//         }
//     }
// }

// impl Build<Label> for dyn Statement {
//     fn build(_pos: TextPosition, _context: &mut ParserContext, param: Label) -> Self::ParseOutputType {

//         Box::new(param)
//     }
// }

// impl Parseable for Goto {
//     type ParseOutputType = Self;
// }

// impl Build<(Name,)> for Goto {
//     fn build(pos: TextPosition, _context: &mut ParserContext, param: (Name,)) -> Self::ParseOutputType {

//         Goto {
//             pos: pos,
//             target: param.0,
//         }
//     }
// }

// impl Build<Goto> for dyn Statement {
//     fn build(_pos: TextPosition, _context: &mut ParserContext, param: Goto) -> Self::ParseOutputType {

//         Box::new(param)
//     }
// }

impl Build<Block> for dyn Statement {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: Block) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for Return {
    type ParseOutputType = Self;
}

impl Build<(Option<<Expression as Parseable>::ParseOutputType>,)> for Return {
    fn build(
        pos: TextPosition, _context: &mut ParserContext,
        param: (Option<<Expression as Parseable>::ParseOutputType>,),
    ) -> Self::ParseOutputType {
        Return::new(pos, param.0)
    }
}

impl Build<Return> for dyn Statement {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: Return) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for LocalVariableDeclaration {
    type ParseOutputType = Self;
}

impl
    Build<(
        Name,
        <Type as Parseable>::ParseOutputType,
        <Expression as Parseable>::ParseOutputType,
    )> for LocalVariableDeclaration
{
    fn build(
        pos: TextPosition, context: &mut ParserContext,
        param: (
            Name,
            <Type as Parseable>::ParseOutputType,
            <Expression as Parseable>::ParseOutputType,
        ),
    ) -> Self::ParseOutputType {

        LocalVariableDeclaration {
            declaration: Declaration::build(pos, context, (param.0, param.1)),
            value: param.2,
        }
    }
}

impl Build<LocalVariableDeclaration> for dyn Statement {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: LocalVariableDeclaration) -> Self::ParseOutputType {
        Box::new(param)
    }
}

impl Parseable for Assignment {
    type ParseOutputType = Self;
}

impl
    Build<(
        <Expression as Parseable>::ParseOutputType,
        <Expression as Parseable>::ParseOutputType,
    )> for Assignment
{
    fn build(
        pos: TextPosition, _context: &mut ParserContext,
        param: (
            <Expression as Parseable>::ParseOutputType,
            <Expression as Parseable>::ParseOutputType,
        ),
    ) -> Self::ParseOutputType {
        Assignment::new(pos, param.0, param.1)
    }
}

impl Build<Assignment> for dyn Statement {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: Assignment) -> Self::ParseOutputType {
        Box::new(param)
    }
}

impl Parseable for ArrayEntryAccessData {
    type ParseOutputType = Self;
}

impl Build<(Option<RWModifier>, Vec<Expression>, Alias)> for ArrayEntryAccessData {
    fn build(
        pos: TextPosition, _context: &mut ParserContext,
        param: (Option<RWModifier>, Vec<Expression>, Alias),
    ) -> Self::ParseOutputType {

        let write = match param.0 {
            Some(RWModifier::ReadModifier(_)) => false,
            Some(RWModifier::WriteModifier(_)) => true,
            None => true,
        };

        ArrayEntryAccessData::new(pos, ((param.2).1).0, param.1, write)
    }
}

impl Parseable for ArrayAccessPatternData {
    type ParseOutputType = Self;
}

impl Build<(Vec<ArrayEntryAccessData>, Expression)> for ArrayAccessPatternData {
    fn build(
        pos: TextPosition, _context: &mut ParserContext,
        param: (Vec<ArrayEntryAccessData>, Expression),
    ) -> Self::ParseOutputType {
        ArrayAccessPatternData::new(pos, param.1, param.0)
    }
}

impl Parseable for ParallelFor {
    type ParseOutputType = Self;
}

impl Build<(Vec<DeclarationListNode>, Vec<ArrayAccessPatternData>, Block)> for ParallelFor {
    fn build(
        pos: TextPosition, _context: &mut ParserContext,
        param: (Vec<DeclarationListNode>, Vec<ArrayAccessPatternData>, Block),
    ) -> Self::ParseOutputType {
        ParallelFor::new(pos, param.0.into_iter().map(|x| ((x.1).0, (x.1).1)).collect(), param.1, param.2)
    }
}

impl Build<ParallelFor> for dyn Statement {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: ParallelFor) -> Self::ParseOutputType {
        Box::new(param)
    }
}

impl Parseable for Label {
    type ParseOutputType = Self;
}

impl Build<(Name,)> for Label {
    fn build(
        pos: TextPosition, _context: &mut ParserContext,
        param: (Name,),
    ) -> Self::ParseOutputType {
        Label::new(pos, param.0)
    }
}

impl Build<Label> for dyn Statement {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: Label) -> Self::ParseOutputType {
        Box::new(param)
    }
}

impl Parseable for Goto {
    type ParseOutputType = Self;
}

impl Build<(Name,)> for Goto {
    fn build(
        pos: TextPosition, _context: &mut ParserContext,
        param: (Name,),
    ) -> Self::ParseOutputType {
        Goto::new(pos, param.0)
    }
}

impl Build<Goto> for dyn Statement {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: Goto) -> Self::ParseOutputType {
        Box::new(param)
    }
}

impl Parseable for Expression {
    type ParseOutputType = Self;
}

fn build_expr<T>(
    pos: TextPosition, 
    context: &mut ParserContext,
    function: BuiltInIdentifier,
    first: T,
    others: Vec<T>,
) -> <Expression as Parseable>::ParseOutputType
where
    Expression: Build<T>,
    T: AstNode,
{
    let mut data = others.into_iter();
    let second: Option<T> = data.next();

    if let Some(sec) = second {
        Expression::apply_operator(
            &pos,
            std::iter::once(first)
                .chain(std::iter::once(sec))
                .chain(data)
                .map(|e| Expression::build(e.pos().clone(), context, e)),
            function,
        )
    } else {
        Expression::build(pos, context, first)
    }
}

impl Build<<Expression as Parseable>::ParseOutputType> for Expression {
    fn build(
        _pos: TextPosition, _context: &mut ParserContext,
        param: <Expression as Parseable>::ParseOutputType,
    ) -> Self::ParseOutputType {
        param
    }
}

impl Build<ExprNodeLevelOr> for Expression {
    fn build(pos: TextPosition, context: &mut ParserContext, param: ExprNodeLevelOr) -> Self::ParseOutputType {
        build_expr(
            pos,
            context,
            BuiltInIdentifier::FunctionOr,
            (param.1).0,
            (param.1).1.into_iter().map(|n| (n.1).0).collect(),
        )
    }
}

impl Build<ExprNodeLevelAnd> for Expression {
    fn build(pos: TextPosition, context: &mut ParserContext, param: ExprNodeLevelAnd) -> Self::ParseOutputType {
        build_expr(
            pos,
            context,
            BuiltInIdentifier::FunctionAnd,
            (param.1).0,
            (param.1).1.into_iter().map(|n| (n.1).0).collect(),
        )
    }
}

impl Build<ExprNodeLevelCmp> for Expression {
    fn build(pos: TextPosition, context: &mut ParserContext, param: ExprNodeLevelCmp) -> Self::ParseOutputType {
        (param.1).1.into_iter().fold(
            Expression::build(pos.clone(), context, *(param.1).0),
            |current, next| {

                let (function, parameter) = match next {
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartEq(node) => (
                        BuiltInIdentifier::FunctionEq,
                        Expression::build(node.0, context, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartNeq(node) => (
                        BuiltInIdentifier::FunctionNeq,
                        Expression::build(node.0, context, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartLeq(node) => (
                        BuiltInIdentifier::FunctionLeq,
                        Expression::build(node.0, context, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartGeq(node) => (
                        BuiltInIdentifier::FunctionGeq,
                        Expression::build(node.0, context, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartLs(node) => (
                        BuiltInIdentifier::FunctionLs,
                        Expression::build(node.0, context, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartGt(node) => (
                        BuiltInIdentifier::FunctionGt,
                        Expression::build(node.0, context, *(node.1).0),
                    ),
                };

                Expression::apply_operator(&pos, vec![current, parameter], function)
            },
        )
    }
}

impl Build<ExprNodeLevelAdd> for Expression {
    fn build(pos: TextPosition, context: &mut ParserContext, param: ExprNodeLevelAdd) -> Self::ParseOutputType {

        let mut start_expr = Expression::build(param.0.clone(), context, (param.1).1);

        if let Some(unary_negation) = (param.1).0 {
            start_expr = Expression::apply_operator(
                &unary_negation.0,
                std::iter::once(start_expr),
                BuiltInIdentifier::FunctionUnaryNeg,
            );
        }
        let others = (param.1).2.into_iter().map(|node| match node {
            ExprNodeLevelAddPart::ExprNodeLevelAddPartAdd(part) => {
                Expression::build(part.0, context, (part.1).0)
            }
            ExprNodeLevelAddPart::ExprNodeLevelAddPartSub(part) => Expression::apply_operator(
                &part.0,
                vec![Expression::build(part.0.clone(), context, (part.1).0)],
                BuiltInIdentifier::FunctionUnaryNeg,
            ),
        }).collect();
        build_expr(
            pos,
            context,
            BuiltInIdentifier::FunctionAdd,
            start_expr,
            others,
        )
    }
}

impl Build<ExprNodeLevelMul> for Expression {
    fn build(pos: TextPosition, context: &mut ParserContext, param: ExprNodeLevelMul) -> Self::ParseOutputType {
        let first = Expression::build(param.0, context, (param.1).0);
        let others = (param.1).1.into_iter().map(|node| match node {
            ExprNodeLevelMulPart::ExprNodeLevelMulPartMul(part) => {
                Expression::build(part.0, context, (part.1).0)
            }
            ExprNodeLevelMulPart::ExprNodeLevelMulPartDiv(part) => Expression::apply_operator(
                &part.0,
                vec![Expression::build(part.0.clone(), context, (part.1).0)],
                BuiltInIdentifier::FunctionUnaryDiv,
            ),
        }).collect();
        build_expr(
            pos,
            context,
            BuiltInIdentifier::FunctionMul,
            first,
            others,
        )
    }
}

impl Build<ExprNodeLevelCall> for Expression {
    fn build(pos: TextPosition, context: &mut ParserContext, param: ExprNodeLevelCall) -> Self::ParseOutputType {
        let call_chain = (param.1).1;
        let start_expr = (param.1).0;
        call_chain
            .into_iter()
            .fold(
                Expression::build(pos, context, start_expr),
                |current, next_call| match next_call {
                    FunctionCallOrIndexAccess::IndexAccessParameters(index_access) => {
                        let mut indices = (index_access.1).0;
                        indices.insert(0, current);
                        Expression::apply_operator(
                            &index_access.0,
                            indices.into_iter(),
                            BuiltInIdentifier::FunctionIndex,
                        )
                    }
                    FunctionCallOrIndexAccess::FunctionCallParameters(function_call) => {
                        Expression::Call(Box::new(FunctionCall {
                            pos: function_call.0,
                            function: current,
                            parameters: (function_call.1).0,
                            result_type_cache: None
                        }))
                    }
                },
            )
    }
}

impl Build<BaseExpr> for Expression {
    fn build(_pos: TextPosition, _context: &mut ParserContext, param: BaseExpr) -> Self::ParseOutputType {
        match param {
            BaseExpr::BracketExpr(node) => (node.1).0,
            BaseExpr::Variable(node) => Expression::Variable(node),
            BaseExpr::Literal(node) => Expression::Literal(node),
        }
    }
}

impl_parse_trait! { Type := TypeNodeView | TypeNodeNoView }

grammar_rule! { TypeRead := Token#Read }

grammar_rule! { TypeWrite := Token#Write }

grammar_rule! { ReadWrite := TypeRead | TypeWrite }

grammar_rule! { TypeNodeView := Token#View TypeNodeNoView }

grammar_rule! { TypeNodeNoView := [ ReadWrite ] PrimitiveTypeNode [ Dimensions ] }

grammar_rule! { Dimensions := Token#SquareBracketOpen { Token#Comma } Token#SquareBracketClose }

grammar_rule! { PrimitiveTypeNode := IntTypeNode | FloatTypeNode }

grammar_rule! { IntTypeNode := Token#Int }

grammar_rule! { FloatTypeNode := Token#Comma }

impl_parse_trait! { Function := Token#Fn Name Token#BracketOpen { DeclarationListNode } Token#BracketClose [ Token#Colon Type ] FunctionImpl }

grammar_rule! { DeclarationListNode := Name Token#Colon Type Token#Comma }

grammar_rule! { FunctionImpl := NativeFunction | Block }

grammar_rule! { NativeFunction := Token#Native Token#Semicolon }

impl Parser for dyn Statement {
    fn is_applicable(stream: &Stream) -> bool {
        If::is_applicable(stream)
            || While::is_applicable(stream)
            || Return::is_applicable(stream)
            || Block::is_applicable(stream)
            || Expression::is_applicable(stream)
            || LocalVariableDeclaration::is_applicable(stream)
            || Goto::is_applicable(stream)
            || Label::is_applicable(stream)
            || ParallelFor::is_applicable(stream)
    }

    fn parse(stream: &mut Stream, context: &mut ParserContext) -> Result<Self::ParseOutputType, CompileError> {

        let pos = stream.pos().clone();

        if If::is_applicable(stream) {
            let statement = If::parse(stream, context)?;
            Ok(<dyn Statement>::build(pos, context, statement))
        } else if While::is_applicable(stream) {
            let statement = While::parse(stream, context)?;
            Ok(<dyn Statement>::build(pos, context, statement))
        } else if Return::is_applicable(stream) {
            let statement = Return::parse(stream, context)?;
            Ok(<dyn Statement>::build(pos, context, statement))
        } else if Block::is_applicable(stream) {
            let statement = Block::parse(stream, context)?;
            Ok(<dyn Statement>::build(pos, context, statement))
        } else if LocalVariableDeclaration::is_applicable(stream) {
            let statement = LocalVariableDeclaration::parse(stream, context)?;
            Ok(<dyn Statement>::build(
                pos, context,
                statement,
            ))
        } else if Label::is_applicable(stream) {
            let statement = Label::parse(stream, context)?;
            Ok(<dyn Statement>::build(pos, context, statement))
        } else if Goto::is_applicable(stream) {
            let statement = Goto::parse(stream, context)?;
            Ok(<dyn Statement>::build(pos, context, statement))
        } else if ParallelFor::is_applicable(stream) {
            let statement = ParallelFor::parse(stream, context)?;
            Ok(<dyn Statement>::build(pos, context, statement))
        } else {
            let expr = Expression::parse(stream, context)?;
            if stream.is_next(&Token::Assign) {
                stream.skip_next(&Token::Assign)?;
                let value = Expression::parse(stream, context)?;
                stream.skip_next(&Token::Semicolon)?;
                let statement = Assignment::build(pos.clone(), context, (expr, value)); 
                return Ok(<dyn Statement>::build(
                    pos, context,
                    statement,
                ));
            } else {
                stream.skip_next(&Token::Semicolon)?;
                let statement = ExpressionNode::build(pos.clone(), context, (expr,)); 
                return Ok(<dyn Statement>::build(
                    pos, context,
                    statement,
                ));
            }
        }
    }
}

impl_parse_trait! { Block := Token#CurlyBracketOpen { Statement } Token#CurlyBracketClose }

impl_parse_trait! { If := Token#If Expression Block }

impl_parse_trait! { While := Token#While Expression Block }

impl_parse_trait! { Return := Token#Return [ Expression ] Token#Semicolon }

impl_parse_trait! { Label := Token#Target Name }

impl_parse_trait! { Goto := Token#Goto Name Token#Semicolon }

grammar_rule! { ExpressionNode := Expression Token#Semicolon }

impl_parse_trait! { LocalVariableDeclaration := Token#Let Name Token#Colon Type Token#Init Expression Token#Semicolon }

grammar_rule! { Alias := Token#As Name }

impl_parse_trait! { ArrayEntryAccessData := [ RWModifier ] Token#This Token#SquareBracketOpen { Expression Token#Comma } Token#SquareBracketClose Alias }

impl_parse_trait! { ArrayAccessPatternData := Token#With { ArrayEntryAccessData Token#Comma } Token#In Expression }

impl_parse_trait! { ParallelFor := Token#PFor { DeclarationListNode } { ArrayAccessPatternData } Block }

grammar_rule! { RWModifier := ReadModifier | WriteModifier }

grammar_rule! { ReadModifier := Token#Read }

grammar_rule! { WriteModifier := Token#Write }

impl_parse_trait! { Expression := ExprNodeLevelOr }

grammar_rule! { ExprNodeLevelOr := ExprNodeLevelAnd { ExprNodeLevelOrPart } }

grammar_rule! { ExprNodeLevelOrPart := Token#OpOr ExprNodeLevelAnd }

grammar_rule! { ExprNodeLevelAnd := ExprNodeLevelCmp { ExprNodeLevelAndPart } }

grammar_rule! { ExprNodeLevelAndPart := Token#OpOr ExprNodeLevelCmp }

grammar_rule! { ExprNodeLevelCmp := ExprNodeLevelAdd { ExprNodeLevelCmpPart } }

grammar_rule! { ExprNodeLevelCmpPart := ExprNodeLevelCmpPartGt | ExprNodeLevelCmpPartLs | ExprNodeLevelCmpPartGeq | ExprNodeLevelCmpPartLeq | ExprNodeLevelCmpPartEq | ExprNodeLevelCmpPartNeq }

grammar_rule! { ExprNodeLevelCmpPartGt := Token#OpGreater ExprNodeLevelAdd }

grammar_rule! { ExprNodeLevelCmpPartLs := Token#OpLess ExprNodeLevelAdd }

grammar_rule! { ExprNodeLevelCmpPartGeq := Token#OpGreaterEq ExprNodeLevelAdd }

grammar_rule! { ExprNodeLevelCmpPartLeq := Token#OpLessEq ExprNodeLevelAdd }

grammar_rule! { ExprNodeLevelCmpPartEq := Token#OpEqual ExprNodeLevelAdd }

grammar_rule! { ExprNodeLevelCmpPartNeq := Token#OpUnequal ExprNodeLevelAdd }

grammar_rule! { box ExprNodeLevelAdd := [ UnaryNegation ] ExprNodeLevelMul { ExprNodeLevelAddPart } }

grammar_rule! { UnaryNegation := Token#OpSubtract }

grammar_rule! { ExprNodeLevelAddPart := ExprNodeLevelAddPartAdd | ExprNodeLevelAddPartSub }

grammar_rule! { ExprNodeLevelAddPartAdd := Token#OpAdd ExprNodeLevelMul }

grammar_rule! { ExprNodeLevelAddPartSub := Token#OpSubtract ExprNodeLevelMul }

grammar_rule! { ExprNodeLevelMul := ExprNodeLevelCall { ExprNodeLevelMulPart } }

grammar_rule! { ExprNodeLevelMulPart := ExprNodeLevelMulPartMul | ExprNodeLevelMulPartDiv }

grammar_rule! { ExprNodeLevelMulPartMul := Token#OpMult ExprNodeLevelCall }

grammar_rule! { ExprNodeLevelMulPartDiv := Token#OpDivide ExprNodeLevelCall }

grammar_rule! { ExprNodeLevelCall := BaseExpr { FunctionCallOrIndexAccess } }

grammar_rule! { FunctionCallOrIndexAccess := IndexAccessParameters | FunctionCallParameters }

grammar_rule! { IndexAccessParameters := Token#SquareBracketOpen { Expression Token#Comma } Token#SquareBracketClose }

grammar_rule! { FunctionCallParameters := Token#BracketOpen { Expression Token#Comma } Token#BracketClose }

grammar_rule! { BaseExpr := Variable | Literal | BracketExpr }

grammar_rule! { BracketExpr := Token#BracketOpen Expression Token#BracketClose }

impl_parse_trait! { Variable := Name }

#[cfg(test)]
use super::super::lexer::lexer::{fragment_lex, lex_str};

#[test]

fn test_parser() {

    let program = "fn test(a: int[, ], b: int, ) {
        pfor c: int, with this[c, ] as d, in a {
            d = b;
        }
    }";

    let ast = Program::parse(&mut lex_str(program)).unwrap();

    assert_eq!(1, ast.items.len());

    let param0 = &ast.items[0].parameters[0];
    assert_eq!(Name::l("a"), *param0.get_name());
    assert_eq!(
        Type::array_type(PrimitiveType::Int, 1, false), 
        *param0.get_type()
    );

    let pfor = ast.items[0].body.as_ref().unwrap()
        .statements().next().unwrap()
        .any()
        .downcast_ref::<ParallelFor>()
        .unwrap();

    let index_var = &pfor.index_variables[0];
    assert_eq!(Name::l("c"), *index_var.get_name());
    assert_eq!(SCALAR_INT, *index_var.get_type());

    let assignment = pfor.body
        .statements().next().unwrap()
        .any()
        .downcast_ref::<Assignment>()
        .unwrap();

    assert_eq!(
        Expression::Variable(Variable {
            pos: TextPosition::NONEXISTING,
            identifier: Identifier::Name(Name::l("d"))
        }),
        assignment.assignee
    );
}

#[test]

fn test_parse_index_expressions() {

    let text = "a[b,]";

    let mut context = ParserContext::new();
    let expr = Expression::parse(&mut fragment_lex(text), &mut context).unwrap();

    assert_eq!(
        Expression::Call(Box::new(FunctionCall {
            pos: TextPosition::NONEXISTING,
            function: Expression::Variable(Variable {
                pos: TextPosition::NONEXISTING,
                identifier: Identifier::BuiltIn(BuiltInIdentifier::FunctionIndex)
            }),
            parameters: vec![
                Expression::Variable(Variable {
                    pos: TextPosition::NONEXISTING,
                    identifier: Identifier::Name(Name::l("a"))
                }),
                Expression::Variable(Variable {
                    pos: TextPosition::NONEXISTING,
                    identifier: Identifier::Name(Name::l("b"))
                })
            ],
            result_type_cache: None
        })),
        expr
    );
}
