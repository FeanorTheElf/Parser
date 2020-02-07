use super::super::lexer::tokens::{ Stream, Token };
use super::super::language::prelude::*;
use super::parser_gen::Flatten;
use super::{ Buildable, Build, Parse };

impl Buildable for Function
{
    type OutputType = Box<Self>;
}

impl Build<(String, Vec<ParameterNode>, TypeNode, FunctionImpl)> for Function
{
    fn build(pos: TextPosition, params: (String, Vec<ParameterNode>, TypeNode, FunctionImpl)) -> Self::OutputType
    {
        unimplemented!()
    }
}

impl Buildable for Block
{
    type OutputType = Self;
}

impl Build<(Vec<Box<dyn Statement>>,)> for Block
{
    fn build(pos: TextPosition, params: (Vec<Box<dyn Statement>>,)) -> Self::OutputType
    {
        unimplemented!()
    }
}

impl Buildable for dyn Statement
{
    type OutputType = Box<Self>;
}

impl Build<ExpressionNode> for dyn Statement
{
    fn build(pos: TextPosition, params: (ExpressionNode,)) -> Self::OutputType
    {
        unimplemented!()
    }
}

impl Buildable for If
{
    type OutputType = Self;
}

impl Build<(Box<dyn Expression>, Block)> for If
{
    fn build(pos: TextPosition, params: (Box<dyn Expression>, Block)) -> Self::OutputType
    {
        unimplemented!()
    }
}

impl Build<If> for dyn Statement
{
    fn build(pos: TextPosition, params: (If,)) -> Self::OutputType
    {
        unimplemented!()
    }
}

impl Buildable for While
{
    type OutputType = Self;
}

impl Build<(Box<dyn Expression>, Block)> for While
{
    fn build(pos: TextPosition, params: (Box<dyn Expression>, Block)) -> Self::OutputType
    {
        unimplemented!()
    }
}

impl Build<While> for dyn Statement
{
    fn build(pos: TextPosition, params: (While,)) -> Self::OutputType
    {
        unimplemented!()
    }
}

impl Build<Block> for dyn Statement
{
    fn build(pos: TextPosition, params: (Block,)) -> Self::OutputType
    {
        unimplemented!()
    }
}

impl Buildable for Return
{
    type OutputType = Self;
}

impl Build<(Box<dyn Expression>,)> for Return
{
    fn build(pos: TextPosition, params: (Box<dyn Expression>,)) -> Self::OutputType
    {
        unimplemented!()
    }
}

impl Build<(Return,)> for dyn Statement
{
    fn build(pos: TextPosition, params: (Return,)) -> Self::OutputType
    {
        unimplemented!()
    }
}

impl Buildable for dyn Expression
{
    type OutputType = Box<Self>;
}

impl_parse!{ Function => Token#Fn identifier Token#BracketOpen { ParameterNode } Token#BracketClose Token#Colon TypeNode FunctionImpl }
grammar_rule!{ ParameterNode => identifier Token#Colon TypeNode }
grammar_rule!{ TypeNode => PrimitiveType { Dimension } }
grammar_rule!{ Dimension => Token#SquareBracketOpen Token#SquareBracketClose }
grammar_rule!{ PrimitiveType => Token#Int }
grammar_rule!{ FunctionImpl => NativeFunction | ImplementedFunction }
grammar_rule!{ NativeFunction => Token#Native Token#Semicolon }
grammar_rule!{ ImplementedFunction => Block }
impl_parse!{ Block => Token#CurlyBracketOpen { Statement } Token#CurlyBracketClose }
impl_parse!{ dyn Statement => If | While | Return | Block | ExpressionNode }
grammar_rule!{ ExpressionNode => Expression Token#Semicolon }
impl_parse!{ If => Token#If Expression Block }
impl_parse!{ While => Token#While Expression Block }
impl_parse!{ Return => Token#Return Expression Token#Semicolon }