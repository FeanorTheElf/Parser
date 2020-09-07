use super::super::language::prelude::*;
use super::super::lexer::tokens::{Stream, Token};
use super::parser_gen::Flatten;
use super::{Build, Parseable, Parser, TopLevelParser};

use super::super::util::dyn_lifetime::*;
use std::cell::RefCell;

impl Parseable for Program {
    type ParseOutputType = Self;
}

impl TopLevelParser for Program {
    fn parse(stream: &mut Stream) -> Result<Program, CompileError> {
        stream.skip_next(&Token::BOF);
        let mut result = Program {
            types: TypeVec::new(),
            items: Vec::new()
        };
        while Function::is_applicable(stream) {
            result.items.push(Box::new(Function::parse(stream, &mut result.types)?));
        }
        stream.skip_next(&Token::EOF);
        return Ok(result);
    }
}

impl Parseable for Type {
    type ParseOutputType = DynRef<RefCell<Type>>;
}

fn create_array_type(param: TypeNodeNoView) -> ArrayType {
    ArrayType {
        dimension: (param.1).1.map(|x| (x.1).0 as usize).unwrap_or(0),
        base: PrimitiveType::Int
    }
}

impl Build<TypeNodeNoView> for Type {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: TypeNodeNoView) -> Self::ParseOutputType {
        match (param.1).0 { PrimitiveTypeNode::IntTypeNode(_) => {}, _ => unimplemented!() };
        types.push(RefCell::from(Type::Array(create_array_type(param))))
    }
}

impl Build<TypeNodeView> for Type {
    fn build(pos: TextPosition, types: &mut TypeVec, param: TypeNodeView) -> Self::ParseOutputType {
        types.push(RefCell::from(Type::View(ViewType {
            base: create_array_type((param.1).0),
            concrete: None
        })))
    }
}

impl Parseable for Name {
    type ParseOutputType = Self;
}

impl Parser for Name {
    fn is_applicable(stream: &Stream) -> bool {

        stream.is_next_identifier()
    }

    fn parse(stream: &mut Stream, types: &mut TypeVec) -> Result<Self::ParseOutputType, CompileError> {

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
    fn build(pos: TextPosition, types: &mut TypeVec, param: Name) -> Self::ParseOutputType {

        Variable {
            pos: pos,
            identifier: Identifier::Name(param),
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

    fn parse(stream: &mut Stream, types: &mut TypeVec) -> Result<Self::ParseOutputType, CompileError> {

        Ok(Literal {
            pos: stream.pos().clone(),
            value: stream.next_literal()?,
        })
    }
}

impl Parseable for Declaration {
    type ParseOutputType = Self;
}

impl Build<(Name, <Type as Parseable>::ParseOutputType)> for Declaration {
    fn build(pos: TextPosition, types: &mut TypeVec, param: (Name, <Type as Parseable>::ParseOutputType)) -> Self::ParseOutputType {

        Declaration {
            pos: pos,
            variable_type: param.1,
            variable: param.0,
        }
    }
}

impl Parseable for Function {
    type ParseOutputType = Self;
}

impl Build<(Name, Vec<DeclarationListNode>, Option<<Type as Parseable>::ParseOutputType>, FunctionImpl)> for Function {
    fn build(
        pos: TextPosition, types: &mut TypeVec,
        param: (Name, Vec<DeclarationListNode>, Option<<Type as Parseable>::ParseOutputType>, FunctionImpl),
    ) -> Self::ParseOutputType {

        let block = if let FunctionImpl::Block(block) = param.3 {

            Some(block)
        } else {

            None
        };

        Function {
            pos: pos,
            identifier: param.0,
            params: param
                .1
                .into_iter()
                .map(|p| Declaration::build(p.0, types, p.1))
                .collect(),
            return_type: param.2,
            body: block,
        }
    }
}

impl Parseable for Block {
    type ParseOutputType = Self;
}

impl Build<(Vec<Box<dyn Statement>>,)> for Block {
    fn build(pos: TextPosition, types: &mut TypeVec, param: (Vec<Box<dyn Statement>>,)) -> Self::ParseOutputType {

        Block {
            pos: pos,
            statements: param.0,
        }
    }
}

impl Parseable for dyn Statement {
    type ParseOutputType = Box<Self>;
}

impl Build<ExpressionNode> for dyn Statement {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: ExpressionNode) -> Self::ParseOutputType {

        Box::new((param.1).0)
    }
}

impl Parseable for If {
    type ParseOutputType = Self;
}

impl Build<(<Expression as Parseable>::ParseOutputType, Block)> for If {
    fn build(
        pos: TextPosition, types: &mut TypeVec,
        param: (<Expression as Parseable>::ParseOutputType, Block),
    ) -> Self::ParseOutputType {

        If {
            pos: pos,
            condition: param.0,
            body: param.1,
        }
    }
}

impl Build<If> for dyn Statement {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: If) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for While {
    type ParseOutputType = Self;
}

impl Build<(<Expression as Parseable>::ParseOutputType, Block)> for While {
    fn build(
        pos: TextPosition, types: &mut TypeVec,
        param: (<Expression as Parseable>::ParseOutputType, Block),
    ) -> Self::ParseOutputType {

        While {
            pos: pos,
            condition: param.0,
            body: param.1,
        }
    }
}

impl Build<While> for dyn Statement {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: While) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for Label {
    type ParseOutputType = Self;
}

impl Build<(Name,)> for Label {
    fn build(pos: TextPosition, types: &mut TypeVec, param: (Name,)) -> Self::ParseOutputType {

        Label {
            pos: pos,
            label: param.0,
        }
    }
}

impl Build<Label> for dyn Statement {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: Label) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for Goto {
    type ParseOutputType = Self;
}

impl Build<(Name,)> for Goto {
    fn build(pos: TextPosition, types: &mut TypeVec, param: (Name,)) -> Self::ParseOutputType {

        Goto {
            pos: pos,
            target: param.0,
        }
    }
}

impl Build<Goto> for dyn Statement {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: Goto) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Build<Block> for dyn Statement {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: Block) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for Return {
    type ParseOutputType = Self;
}

impl Build<(Option<<Expression as Parseable>::ParseOutputType>,)> for Return {
    fn build(
        pos: TextPosition, types: &mut TypeVec,
        param: (Option<<Expression as Parseable>::ParseOutputType>,),
    ) -> Self::ParseOutputType {

        Return {
            pos: pos,
            value: param.0,
        }
    }
}

impl Build<Return> for dyn Statement {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: Return) -> Self::ParseOutputType {

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
        Option<<Expression as Parseable>::ParseOutputType>,
    )> for LocalVariableDeclaration
{
    fn build(
        pos: TextPosition, types: &mut TypeVec,
        param: (
            Name,
            <Type as Parseable>::ParseOutputType,
            Option<<Expression as Parseable>::ParseOutputType>,
        ),
    ) -> Self::ParseOutputType {

        LocalVariableDeclaration {
            declaration: Declaration::build(pos, types, (param.0, param.1)),
            value: param.2,
        }
    }
}

impl Build<LocalVariableDeclaration> for dyn Statement {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: LocalVariableDeclaration) -> Self::ParseOutputType {

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
        pos: TextPosition, types: &mut TypeVec,
        param: (
            <Expression as Parseable>::ParseOutputType,
            <Expression as Parseable>::ParseOutputType,
        ),
    ) -> Self::ParseOutputType {

        Assignment {
            pos: pos,
            assignee: param.0,
            value: param.1,
        }
    }
}

impl Build<Assignment> for dyn Statement {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: Assignment) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for ArrayEntryAccess {
    type ParseOutputType = Self;
}

impl Build<(Option<RWModifier>, Vec<Expression>, Option<Alias>)> for ArrayEntryAccess {
    fn build(
        pos: TextPosition, types: &mut TypeVec,
        param: (Option<RWModifier>, Vec<Expression>, Option<Alias>),
    ) -> Self::ParseOutputType {

        let write = match param.0 {
            Some(RWModifier::ReadModifier(_)) => false,
            Some(RWModifier::WriteModifier(_)) => true,
            None => true,
        };

        ArrayEntryAccess::new(pos, param.1, param.2.map(|alias| (alias.1).0), write)
    }
}

impl Parseable for ArrayAccessPattern {
    type ParseOutputType = Self;
}

impl Build<(Vec<ArrayEntryAccess>, Expression)> for ArrayAccessPattern {
    fn build(
        pos: TextPosition, types: &mut TypeVec,
        param: (Vec<ArrayEntryAccess>, Expression),
    ) -> Self::ParseOutputType {

        ArrayAccessPattern {
            pos: pos,
            array: param.1,
            entry_accesses: param.0,
        }
    }
}

impl Parseable for ParallelFor {
    type ParseOutputType = Self;
}

impl Build<(Vec<DeclarationListNode>, Vec<ArrayAccessPattern>, Block)> for ParallelFor {
    fn build(
        pos: TextPosition, types: &mut TypeVec,
        param: (Vec<DeclarationListNode>, Vec<ArrayAccessPattern>, Block),
    ) -> Self::ParseOutputType {

        ParallelFor {
            pos: pos,
            access_pattern: param.1,
            index_variables: param
                .0
                .into_iter()
                .map(|node| Declaration::build(node.0, types, node.1))
                .collect(),
            body: param.2,
        }
    }
}

impl Build<ParallelFor> for dyn Statement {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: ParallelFor) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for Expression {
    type ParseOutputType = Self;
}

fn build_function_call(
    pos: TextPosition,
    function: BuiltInIdentifier,
    params: Vec<<Expression as Parseable>::ParseOutputType>,
) -> Expression {

    Expression::Call(Box::new(FunctionCall {
        pos: pos.clone(),
        function: Expression::Variable(Variable {
            pos: pos,
            identifier: Identifier::BuiltIn(function),
        }),
        parameters: params,
    }))
}

fn build_expr<T>(
    pos: TextPosition, 
    types: &mut TypeVec,
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

        build_function_call(
            pos,
            function,
            std::iter::once(first)
                .chain(std::iter::once(sec))
                .chain(data)
                .map(|e| Expression::build(e.pos().clone(), types, e))
                .collect(),
        )
    } else {

        Expression::build(pos, types, first)
    }
}

impl Build<<Expression as Parseable>::ParseOutputType> for Expression {
    fn build(
        _pos: TextPosition, _types: &mut TypeVec,
        param: <Expression as Parseable>::ParseOutputType,
    ) -> Self::ParseOutputType {

        param
    }
}

impl Build<ExprNodeLevelOr> for Expression {
    fn build(pos: TextPosition, types: &mut TypeVec, param: ExprNodeLevelOr) -> Self::ParseOutputType {

        build_expr(
            pos,
            types,
            BuiltInIdentifier::FunctionOr,
            (param.1).0,
            (param.1).1.into_iter().map(|n| (n.1).0).collect(),
        )
    }
}

impl Build<ExprNodeLevelAnd> for Expression {
    fn build(pos: TextPosition, types: &mut TypeVec, param: ExprNodeLevelAnd) -> Self::ParseOutputType {

        build_expr(
            pos,
            types,
            BuiltInIdentifier::FunctionAnd,
            (param.1).0,
            (param.1).1.into_iter().map(|n| (n.1).0).collect(),
        )
    }
}

impl Build<ExprNodeLevelCmp> for Expression {
    fn build(pos: TextPosition, types: &mut TypeVec, param: ExprNodeLevelCmp) -> Self::ParseOutputType {

        (param.1).1.into_iter().fold(
            Expression::build(pos.clone(), types, *(param.1).0),
            |current, next| {

                let (function, parameter) = match next {
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartEq(node) => (
                        BuiltInIdentifier::FunctionEq,
                        Expression::build(node.0, types, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartNeq(node) => (
                        BuiltInIdentifier::FunctionNeq,
                        Expression::build(node.0, types, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartLeq(node) => (
                        BuiltInIdentifier::FunctionLeq,
                        Expression::build(node.0, types, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartGeq(node) => (
                        BuiltInIdentifier::FunctionGeq,
                        Expression::build(node.0, types, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartLs(node) => (
                        BuiltInIdentifier::FunctionLs,
                        Expression::build(node.0, types, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartGt(node) => (
                        BuiltInIdentifier::FunctionGt,
                        Expression::build(node.0, types, *(node.1).0),
                    ),
                };

                build_function_call(pos.clone(), function, vec![current, parameter])
            },
        )
    }
}

impl Build<ExprNodeLevelAdd> for Expression {
    fn build(pos: TextPosition, types: &mut TypeVec, param: ExprNodeLevelAdd) -> Self::ParseOutputType {

        let mut start_expr = Expression::build(param.0.clone(), types, (param.1).1);

        if let Some(unary_negation) = (param.1).0 {

            start_expr = build_function_call(
                unary_negation.0,
                BuiltInIdentifier::FunctionUnaryNeg,
                vec![start_expr],
            );
        }
        let others = (param.1).2.into_iter().map(|node| match node {
            ExprNodeLevelAddPart::ExprNodeLevelAddPartAdd(part) => {
                Expression::build(part.0, types, (part.1).0)
            }
            ExprNodeLevelAddPart::ExprNodeLevelAddPartSub(part) => build_function_call(
                part.0.clone(),
                BuiltInIdentifier::FunctionUnaryNeg,
                vec![Expression::build(part.0, types, (part.1).0)],
            ),
        }).collect();
        build_expr(
            pos,
            types,
            BuiltInIdentifier::FunctionAdd,
            start_expr,
            others,
        )
    }
}

impl Build<ExprNodeLevelMul> for Expression {
    fn build(pos: TextPosition, types: &mut TypeVec, param: ExprNodeLevelMul) -> Self::ParseOutputType {

        let first = Expression::build(param.0, types, (param.1).0);
        let others = (param.1).1.into_iter().map(|node| match node {
            ExprNodeLevelMulPart::ExprNodeLevelMulPartMul(part) => {
                Expression::build(part.0, types, (part.1).0)
            }
            ExprNodeLevelMulPart::ExprNodeLevelMulPartDiv(part) => build_function_call(
                part.0.clone(),
                BuiltInIdentifier::FunctionUnaryDiv,
                vec![Expression::build(part.0, types, (part.1).0)],
            ),
        }).collect();
        build_expr(
            pos,
            types,
            BuiltInIdentifier::FunctionMul,
            first,
            others,
        )
    }
}

impl Build<ExprNodeLevelCall> for Expression {
    fn build(pos: TextPosition, types: &mut TypeVec, param: ExprNodeLevelCall) -> Self::ParseOutputType {

        let call_chain = (param.1).1;

        let start_expr = (param.1).0;

        call_chain
            .into_iter()
            .fold(
                Expression::build(pos, types, start_expr),
                |current, next_call| match next_call {
                    FunctionCallOrIndexAccess::IndexAccessParameters(index_access) => {

                        let mut indices = (index_access.1).0;

                        indices.insert(0, current);

                        build_function_call(
                            index_access.0,
                            BuiltInIdentifier::FunctionIndex,
                            indices,
                        )
                    }
                    FunctionCallOrIndexAccess::FunctionCallParameters(function_call) => {
                        Expression::Call(Box::new(FunctionCall {
                            pos: function_call.0,
                            function: current,
                            parameters: (function_call.1).0,
                        }))
                    }
                },
            )
    }
}

impl Build<BaseExpr> for Expression {
    fn build(_pos: TextPosition, types: &mut TypeVec, param: BaseExpr) -> Self::ParseOutputType {

        match param {
            BaseExpr::BracketExpr(node) => (node.1).0,
            BaseExpr::Variable(node) => Expression::Variable(node),
            BaseExpr::Literal(node) => Expression::Literal(node),
        }
    }
}

impl_parse! { Type := TypeNodeView | TypeNodeNoView }

grammar_rule! { TypeNodeView := Token#View TypeNodeNoView }

grammar_rule! { TypeNodeNoView := PrimitiveTypeNode [ Dimensions ] }

grammar_rule! { Dimensions := Token#SquareBracketOpen { Token#Comma } Token#SquareBracketClose }

grammar_rule! { PrimitiveTypeNode := IntTypeNode | FloatTypeNode }

grammar_rule! { IntTypeNode := Token#Int }

grammar_rule! { FloatTypeNode := Token#Comma }

impl_parse! { Function := Token#Fn Name Token#BracketOpen { DeclarationListNode } Token#BracketClose [ Token#Colon Type ] FunctionImpl }

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

    fn parse(stream: &mut Stream, types: &mut TypeVec) -> Result<Self::ParseOutputType, CompileError> {

        let pos = stream.pos().clone();

        if If::is_applicable(stream) {
            let statement = If::parse(stream, types)?;
            Ok(Statement::build(pos, types, statement))
        } else if While::is_applicable(stream) {
            let statement = While::parse(stream, types)?;
            Ok(Statement::build(pos, types, statement))
        } else if Return::is_applicable(stream) {
            let statement = Return::parse(stream, types)?;
            Ok(Statement::build(pos, types, statement))
        } else if Block::is_applicable(stream) {
            let statement = Block::parse(stream, types)?;
            Ok(Statement::build(pos, types, statement))
        } else if LocalVariableDeclaration::is_applicable(stream) {
            let statement = LocalVariableDeclaration::parse(stream, types)?;
            Ok(Statement::build(
                pos, types,
                statement,
            ))
        } else if Label::is_applicable(stream) {
            let statement = Label::parse(stream, types)?;
            Ok(Statement::build(pos, types, statement))
        } else if Goto::is_applicable(stream) {
            let statement = Goto::parse(stream, types)?;
            Ok(Statement::build(pos, types, statement))
        } else if ParallelFor::is_applicable(stream) {
            let statement = ParallelFor::parse(stream, types)?;
            Ok(Statement::build(pos, types, statement))
        } else {
            let expr = Expression::parse(stream, types)?;
            if stream.is_next(&Token::Assign) {
                stream.skip_next(&Token::Assign)?;
                let value = Expression::parse(stream, types)?;
                stream.skip_next(&Token::Semicolon)?;
                let statement = Assignment::build(pos.clone(), types, (expr, value)); 
                return Ok(Statement::build(
                    pos, types,
                    statement,
                ));
            } else {
                stream.skip_next(&Token::Semicolon)?;
                let statement = ExpressionNode::build(pos.clone(), types, (expr,)); 
                return Ok(Statement::build(
                    pos, types,
                    statement,
                ));
            }
        }
    }
}

impl_parse! { Block := Token#CurlyBracketOpen { Statement } Token#CurlyBracketClose }

impl_parse! { If := Token#If Expression Block }

impl_parse! { While := Token#While Expression Block }

impl_parse! { Return := Token#Return [ Expression ] Token#Semicolon }

impl_parse! { Label := Token#Target Name }

impl_parse! { Goto := Token#Goto Name Token#Semicolon }

grammar_rule! { ExpressionNode := Expression Token#Semicolon }

impl_parse! { LocalVariableDeclaration := Token#Let Name Token#Colon Type [Token#Assign Expression] Token#Semicolon }

grammar_rule! { Alias := Token#As Name }

impl_parse! { ArrayEntryAccess := [ RWModifier ] Token#This Token#SquareBracketOpen { Expression Token#Comma } Token#SquareBracketClose [ Alias ] }

impl_parse! { ArrayAccessPattern := Token#With { ArrayEntryAccess Token#Comma } Token#In Expression }

impl_parse! { ParallelFor := Token#PFor { DeclarationListNode } { ArrayAccessPattern } Block }

grammar_rule! { RWModifier := ReadModifier | WriteModifier }

grammar_rule! { ReadModifier := Token#Read }

grammar_rule! { WriteModifier := Token#Write }

impl_parse! { Expression := ExprNodeLevelOr }

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

impl_parse! { Variable := Name }

#[cfg(test)]
use super::super::language::position::NONEXISTING;
#[cfg(test)]
use super::super::lexer::lexer::{fragment_lex, lex_str};

#[test]

fn test_parser() {

    let program = "fn test(a: int[, ], b: int,) {
        pfor c: int, with this[c, ] as d, in a {
            d = b;
        }
    }";

    let ast = Program::parse(&mut lex_str(program)).unwrap();

    assert_eq!(1, ast.items.len());
    assert_eq!(3, ast.types.len());

    assert_eq!(
        Declaration {
            pos: NONEXISTING,
            variable: Name::l("a"),
            variable_type: ast.types.at(0)
        },
        ast.items[0].params[0]
    );

    assert_eq!(None, ast.items[0].return_type);

    let pfor = ast.items[0].body.as_ref().unwrap().statements[0]
        .dynamic()
        .downcast_ref::<ParallelFor>()
        .unwrap();

    assert_eq!(
        Declaration {
            pos: NONEXISTING,
            variable: Name::l("c"),
            variable_type: ast.types.at(2)
        },
        pfor.index_variables[0]
    );

    let assignment = pfor.body.statements[0]
        .dynamic()
        .downcast_ref::<Assignment>()
        .unwrap();

    assert_eq!(
        Expression::Variable(Variable {
            pos: NONEXISTING,
            identifier: Identifier::Name(Name::l("d"))
        }),
        assignment.assignee
    );
}

#[test]

fn test_parse_index_expressions() {

    let text = "a[b,]";

    let expr = Expression::parse(&mut fragment_lex(text), &mut TypeVec::new()).unwrap();

    assert_eq!(
        Expression::Call(Box::new(FunctionCall {
            pos: NONEXISTING,
            function: Expression::Variable(Variable {
                pos: NONEXISTING,
                identifier: Identifier::BuiltIn(BuiltInIdentifier::FunctionIndex)
            }),
            parameters: vec![
                Expression::Variable(Variable {
                    pos: NONEXISTING,
                    identifier: Identifier::Name(Name::l("a"))
                }),
                Expression::Variable(Variable {
                    pos: NONEXISTING,
                    identifier: Identifier::Name(Name::l("b"))
                })
            ],
        })),
        expr
    );
}
