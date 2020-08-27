use super::super::language::prelude::*;
use super::super::lexer::tokens::{Stream, Token};
use super::parser_gen::Flatten;
use super::{Build, Parseable, Parser};

impl Parseable for Program {
    type ParseOutputType = Self;
}

impl Build<(Vec<Function>,)> for Program {
    fn build(_pos: TextPosition, param: (Vec<Function>,)) -> Self::ParseOutputType {

        Program {
            items: param.0.into_iter().map(&Box::new).collect(),
        }
    }
}

impl Parseable for Type {
    type ParseOutputType = Self;
}

impl Build<TypeNodeNoView> for Type {
    fn build(_pos: TextPosition, param: TypeNodeNoView) -> Self::ParseOutputType {

        if let Some(dimensions) = (param.1).1 {

            Type::Array(PrimitiveType::Int, (dimensions.1).0)
        } else {

            Type::Primitive(PrimitiveType::Int)
        }
    }
}

impl Build<TypeNodeView> for Type {
    fn build(pos: TextPosition, param: TypeNodeView) -> Self::ParseOutputType {

        let view_count = (param.1).0 + 1;

        let mut result = Type::build(pos, (param.1).1);

        for _i in 0..view_count {

            result = Type::View(Box::new(result));
        }

        return result;
    }
}

impl Parseable for Name {
    type ParseOutputType = Self;
}

impl Parser for Name {
    fn is_applicable(stream: &Stream) -> bool {

        stream.is_next_identifier()
    }

    fn parse(stream: &mut Stream) -> Result<Self::ParseOutputType, CompileError> {

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
    fn build(pos: TextPosition, param: Name) -> Self::ParseOutputType {

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

    fn parse(stream: &mut Stream) -> Result<Self::ParseOutputType, CompileError> {

        Ok(Literal {
            pos: stream.pos().clone(),
            value: stream.next_literal()?,
        })
    }
}

impl Parseable for Declaration {
    type ParseOutputType = Self;
}

impl Build<(Name, Type)> for Declaration {
    fn build(pos: TextPosition, param: (Name, Type)) -> Self::ParseOutputType {

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

impl Build<(Name, Vec<DeclarationListNode>, Option<Type>, FunctionImpl)> for Function {
    fn build(
        pos: TextPosition,
        param: (Name, Vec<DeclarationListNode>, Option<Type>, FunctionImpl),
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
                .map(|p| Declaration::build(p.0, p.1))
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
    fn build(pos: TextPosition, param: (Vec<Box<dyn Statement>>,)) -> Self::ParseOutputType {

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
    fn build(_pos: TextPosition, param: ExpressionNode) -> Self::ParseOutputType {

        Box::new((param.1).0)
    }
}

impl Parseable for If {
    type ParseOutputType = Self;
}

impl Build<(<Expression as Parseable>::ParseOutputType, Block)> for If {
    fn build(
        pos: TextPosition,
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
    fn build(_pos: TextPosition, param: If) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for While {
    type ParseOutputType = Self;
}

impl Build<(<Expression as Parseable>::ParseOutputType, Block)> for While {
    fn build(
        pos: TextPosition,
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
    fn build(_pos: TextPosition, param: While) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for Label {
    type ParseOutputType = Self;
}

impl Build<(Name,)> for Label {
    fn build(pos: TextPosition, param: (Name,)) -> Self::ParseOutputType {

        Label {
            pos: pos,
            label: param.0,
        }
    }
}

impl Build<Label> for dyn Statement {
    fn build(_pos: TextPosition, param: Label) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for Goto {
    type ParseOutputType = Self;
}

impl Build<(Name,)> for Goto {
    fn build(pos: TextPosition, param: (Name,)) -> Self::ParseOutputType {

        Goto {
            pos: pos,
            target: param.0,
        }
    }
}

impl Build<Goto> for dyn Statement {
    fn build(_pos: TextPosition, param: Goto) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Build<Block> for dyn Statement {
    fn build(_pos: TextPosition, param: Block) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for Return {
    type ParseOutputType = Self;
}

impl Build<(Option<<Expression as Parseable>::ParseOutputType>,)> for Return {
    fn build(
        pos: TextPosition,
        param: (Option<<Expression as Parseable>::ParseOutputType>,),
    ) -> Self::ParseOutputType {

        Return {
            pos: pos,
            value: param.0,
        }
    }
}

impl Build<Return> for dyn Statement {
    fn build(_pos: TextPosition, param: Return) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for LocalVariableDeclaration {
    type ParseOutputType = Self;
}

impl
    Build<(
        Name,
        Type,
        Option<<Expression as Parseable>::ParseOutputType>,
    )> for LocalVariableDeclaration
{
    fn build(
        pos: TextPosition,
        param: (
            Name,
            Type,
            Option<<Expression as Parseable>::ParseOutputType>,
        ),
    ) -> Self::ParseOutputType {

        LocalVariableDeclaration {
            declaration: Declaration::build(pos, (param.0, param.1)),
            value: param.2,
        }
    }
}

impl Build<LocalVariableDeclaration> for dyn Statement {
    fn build(_pos: TextPosition, param: LocalVariableDeclaration) -> Self::ParseOutputType {

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
        pos: TextPosition,
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
    fn build(_pos: TextPosition, param: Assignment) -> Self::ParseOutputType {

        Box::new(param)
    }
}

impl Parseable for ArrayEntryAccess {
    type ParseOutputType = Self;
}

impl Build<(Option<RWModifier>, Vec<Expression>, Option<Alias>)> for ArrayEntryAccess {
    fn build(
        pos: TextPosition,
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
        pos: TextPosition,
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
        pos: TextPosition,
        param: (Vec<DeclarationListNode>, Vec<ArrayAccessPattern>, Block),
    ) -> Self::ParseOutputType {

        ParallelFor {
            pos: pos,
            access_pattern: param.1,
            index_variables: param
                .0
                .into_iter()
                .map(|node| Declaration::build(node.0, node.1))
                .collect(),
            body: param.2,
        }
    }
}

impl Build<ParallelFor> for dyn Statement {
    fn build(_pos: TextPosition, param: ParallelFor) -> Self::ParseOutputType {

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

fn build_expr<T, I>(
    pos: TextPosition,
    function: BuiltInIdentifier,
    first: T,
    mut others: I,
) -> <Expression as Parseable>::ParseOutputType
where
    Expression: Build<T>,
    I: Iterator<Item = T>,
    T: AstNode,
{

    let second: Option<T> = others.next();

    if let Some(sec) = second {

        build_function_call(
            pos,
            function,
            std::iter::once(first)
                .chain(std::iter::once(sec))
                .chain(others)
                .map(|e| Expression::build(e.pos().clone(), e))
                .collect(),
        )
    } else {

        Expression::build(pos, first)
    }
}

impl Build<<Expression as Parseable>::ParseOutputType> for Expression {
    fn build(
        _pos: TextPosition,
        param: <Expression as Parseable>::ParseOutputType,
    ) -> Self::ParseOutputType {

        param
    }
}

impl Build<ExprNodeLevelOr> for Expression {
    fn build(pos: TextPosition, param: ExprNodeLevelOr) -> Self::ParseOutputType {

        build_expr(
            pos,
            BuiltInIdentifier::FunctionOr,
            (param.1).0,
            (param.1).1.into_iter().map(|n| (n.1).0),
        )
    }
}

impl Build<ExprNodeLevelAnd> for Expression {
    fn build(pos: TextPosition, param: ExprNodeLevelAnd) -> Self::ParseOutputType {

        build_expr(
            pos,
            BuiltInIdentifier::FunctionAnd,
            (param.1).0,
            (param.1).1.into_iter().map(|n| (n.1).0),
        )
    }
}

impl Build<ExprNodeLevelCmp> for Expression {
    fn build(pos: TextPosition, param: ExprNodeLevelCmp) -> Self::ParseOutputType {

        (param.1).1.into_iter().fold(
            Expression::build(pos.clone(), *(param.1).0),
            |current, next| {

                let (function, parameter) = match next {
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartEq(node) => (
                        BuiltInIdentifier::FunctionEq,
                        Expression::build(node.0, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartNeq(node) => (
                        BuiltInIdentifier::FunctionNeq,
                        Expression::build(node.0, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartLeq(node) => (
                        BuiltInIdentifier::FunctionLeq,
                        Expression::build(node.0, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartGeq(node) => (
                        BuiltInIdentifier::FunctionGeq,
                        Expression::build(node.0, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartLs(node) => (
                        BuiltInIdentifier::FunctionLs,
                        Expression::build(node.0, *(node.1).0),
                    ),
                    ExprNodeLevelCmpPart::ExprNodeLevelCmpPartGt(node) => (
                        BuiltInIdentifier::FunctionGt,
                        Expression::build(node.0, *(node.1).0),
                    ),
                };

                build_function_call(pos.clone(), function, vec![current, parameter])
            },
        )
    }
}

impl Build<ExprNodeLevelAdd> for Expression {
    fn build(pos: TextPosition, param: ExprNodeLevelAdd) -> Self::ParseOutputType {

        let mut start_expr = Expression::build(param.0.clone(), (param.1).1);

        if let Some(unary_negation) = (param.1).0 {

            start_expr = build_function_call(
                unary_negation.0,
                BuiltInIdentifier::FunctionUnaryNeg,
                vec![start_expr],
            );
        }

        build_expr(
            pos,
            BuiltInIdentifier::FunctionAdd,
            start_expr,
            (param.1).2.into_iter().map(|node| match node {
                ExprNodeLevelAddPart::ExprNodeLevelAddPartAdd(part) => {
                    Expression::build(part.0, (part.1).0)
                }
                ExprNodeLevelAddPart::ExprNodeLevelAddPartSub(part) => build_function_call(
                    part.0.clone(),
                    BuiltInIdentifier::FunctionUnaryNeg,
                    vec![Expression::build(part.0, (part.1).0)],
                ),
            }),
        )
    }
}

impl Build<ExprNodeLevelMul> for Expression {
    fn build(pos: TextPosition, param: ExprNodeLevelMul) -> Self::ParseOutputType {

        build_expr(
            pos,
            BuiltInIdentifier::FunctionMul,
            Expression::build(param.0, (param.1).0),
            (param.1).1.into_iter().map(|node| match node {
                ExprNodeLevelMulPart::ExprNodeLevelMulPartMul(part) => {
                    Expression::build(part.0, (part.1).0)
                }
                ExprNodeLevelMulPart::ExprNodeLevelMulPartDiv(part) => build_function_call(
                    part.0.clone(),
                    BuiltInIdentifier::FunctionUnaryDiv,
                    vec![Expression::build(part.0, (part.1).0)],
                ),
            }),
        )
    }
}

impl Build<ExprNodeLevelCall> for Expression {
    fn build(pos: TextPosition, param: ExprNodeLevelCall) -> Self::ParseOutputType {

        let call_chain = (param.1).1;

        let start_expr = (param.1).0;

        call_chain
            .into_iter()
            .fold(
                Expression::build(pos, start_expr),
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
    fn build(_pos: TextPosition, param: BaseExpr) -> Self::ParseOutputType {

        match param {
            BaseExpr::BracketExpr(node) => (node.1).0,
            BaseExpr::Variable(node) => Expression::Variable(node),
            BaseExpr::Literal(node) => Expression::Literal(node),
        }
    }
}

impl_parse! { Program := Token#BOF { Function } Token#EOF }

impl_parse! { Type := TypeNodeView | TypeNodeNoView }

grammar_rule! { TypeNodeView := Token#View { Token#View } TypeNodeNoView }

grammar_rule! { TypeNodeNoView := PrimitiveTypeNode [ Dimensions ] }

grammar_rule! { Dimensions := Token#SquareBracketOpen { Token#Comma } Token#SquareBracketClose }

grammar_rule! { PrimitiveTypeNode := Token#Int }

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

    fn parse(stream: &mut Stream) -> Result<Self::ParseOutputType, CompileError> {

        let pos = stream.pos().clone();

        if If::is_applicable(stream) {

            Ok(Statement::build(pos, If::parse(stream)?))
        } else if While::is_applicable(stream) {

            Ok(Statement::build(pos, While::parse(stream)?))
        } else if Return::is_applicable(stream) {

            Ok(Statement::build(pos, Return::parse(stream)?))
        } else if Block::is_applicable(stream) {

            Ok(Statement::build(pos, Block::parse(stream)?))
        } else if LocalVariableDeclaration::is_applicable(stream) {

            Ok(Statement::build(
                pos,
                LocalVariableDeclaration::parse(stream)?,
            ))
        } else if Label::is_applicable(stream) {

            Ok(Statement::build(pos, Label::parse(stream)?))
        } else if Goto::is_applicable(stream) {

            Ok(Statement::build(pos, Goto::parse(stream)?))
        } else if ParallelFor::is_applicable(stream) {

            Ok(Statement::build(pos, ParallelFor::parse(stream)?))
        } else {

            let expr = Expression::parse(stream)?;

            if stream.is_next(&Token::Assign) {

                stream.skip_next(&Token::Assign)?;

                let value = Expression::parse(stream)?;

                stream.skip_next(&Token::Semicolon)?;

                return Ok(Statement::build(
                    pos.clone(),
                    Assignment::build(pos, (expr, value)),
                ));
            } else {

                stream.skip_next(&Token::Semicolon)?;

                return Ok(Statement::build(
                    pos.clone(),
                    ExpressionNode::build(pos, (expr,)),
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
use super::super::lexer::lexer::{fragment_lex, lex};

#[test]

fn test_parser() {

    let program = "fn test(a: int[, ], b: int,) {
        pfor c: int, with this[c, ] as d, in a {
            d = b;
        }
    }";

    let ast = Program::parse(&mut lex(program)).unwrap();

    assert_eq!(1, ast.items.len());

    assert_eq!(
        Declaration {
            pos: NONEXISTING,
            variable: Name::l("a"),
            variable_type: Type::Array(PrimitiveType::Int, 1)
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
            variable_type: Type::Primitive(PrimitiveType::Int)
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

    let expr = Expression::parse(&mut fragment_lex(text)).unwrap();

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
