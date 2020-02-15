use super::super::parser::prelude::*;
use super::scope::*;

use std::collections::{ HashMap, BTreeSet };

struct RenameTransformer<'a>
{
    renaming: &'a HashMap<Identifier, Identifier>
}

impl<'a> Transformer for RenameTransformer<'a>
{
    fn before(&mut self, _node: &dyn Node)
    {
    }

    fn after(&mut self, _node: &dyn Node)
    {
    }

    fn transform_stmt(&mut self, node: Box<dyn StmtNode>) -> Box<dyn StmtNode>
    {
        match cast::<dyn StmtNode, VariableDeclarationNode>(node) {
            Ok(mut decl) => {
                if let Some(new_name) = self.renaming.get(&decl.ident) {
                    decl.ident = new_name.clone();
                }
                decl.transform(self);
                decl
            },
            Err(mut node) => {
                node.transform(self);
                node
            }
        }
    }

    fn transform_expr(&mut self, node: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
    {
        match cast::<dyn UnaryExprNode, VariableNode>(node) {
            Ok(mut var) => {
                if let Some(new_name) = self.renaming.get(&var.identifier) {
                    var.identifier = new_name.clone();
                }
                var
            },
            Err(mut node) => {
                node.transform(self);
                node
            }
        }
    }
}

struct TransformReturnTransformer<'a>
{
    result_name: &'a Identifier
}

impl<'a> Transformer for TransformReturnTransformer<'a>
{
    fn before(&mut self, _node: &dyn Node)
    {
    }

    fn after(&mut self, _node: &dyn Node)
    {
    }

    fn transform_stmt(&mut self, node: Box<dyn StmtNode>) -> Box<dyn StmtNode>
    {
        match cast::<dyn StmtNode, ReturnNode>(node) {
            Ok(ret) => {
                let pos = ret.get_annotation().clone();
                Box::new(AssignmentNode::new(pos.clone(), 
                    Box::new(ExprNodeLvlOr::new(pos.clone(),
                        Box::new(ExprNodeLvlAnd::new(pos.clone(),
                            Box::new(ExprNodeLvlCmp::new(pos.clone(),
                                Box::new(ExprNodeLvlAdd::new(pos.clone(),
                                    Box::new(ExprNodeLvlMult::new(pos.clone(),
                                        Box::new(ExprNodeLvlIndex::new(pos.clone(),
                                            Box::new(VariableNode::new(pos, self.result_name.clone())),
                                        vec![])),
                                    vec![])),
                                vec![])),
                            vec![])),
                        vec![])),
                    vec![])),
                ret.expr))
                // goto node
            },
            Err(mut node) => {
                node.transform(self);
                node
            }
        }
    }

    fn transform_expr(&mut self, node: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
    {
        node
    }
}

struct InlineCallsInTopLevelExpressionsTransformer<'a, I>
    where I: Iterator<Item = (Identifier, Identifier)>
{
    disjunct_names_provider: &'a mut I,
    inline_calls: &'a mut Vec<(Identifier, Box<FunctionCallNode>)>
}

impl<'a, I> Transformer for InlineCallsInTopLevelExpressionsTransformer<'a, I>
    where I: Iterator<Item = (Identifier, Identifier)>
{
    fn before(&mut self, _node: &dyn Node)
    {
    }

    fn after(&mut self, _node: &dyn Node)
    {
    }

    fn transform_stmt(&mut self, node: Box<dyn StmtNode>) -> Box<dyn StmtNode>
    {
        // do not transform child statements recursivly, as InlineCallsInStmtsTransformer will process them.
        // Otherwise, all function calls in the body of an if would be inlined before the if
        node
    }

    fn transform_expr(&mut self, node: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
    {
        match cast::<dyn UnaryExprNode, FunctionCallNode>(node) {
            Ok(call) => {
                let (_, new_name) = self.disjunct_names_provider.next().unwrap();
                let result = Box::new(VariableNode::new(call.get_annotation().clone(), new_name.clone()));
                self.inline_calls.push((new_name, call));
                result
            },
            Err(mut node) => {
                node.transform(self);
                node
            }
        }
    }
}

struct InlineCallsTransformer<'a>
{
    scope_stack: ScopeStack,
    inlinable_functions: &'a [FunctionNode]
}

impl<'a> InlineCallsTransformer<'a>
{
    fn new(inlinable_functions: &'a [FunctionNode]) -> InlineCallsTransformer
    {
        InlineCallsTransformer {
            scope_stack: ScopeStack::new(inlinable_functions),
            inlinable_functions: inlinable_functions
        }
    }

    fn find_variable_names<'b>(block: &'b BlockNode) -> BTreeSet<&'b Identifier>
    {
        let mut result = BTreeSet::new();
        block.iterate(&mut |node: &'b dyn Node| {
            if let Some(decl) = node.dynamic().downcast_ref::<VariableDeclarationNode>() {
                result.insert(&decl.ident);
            } else if let Some(var) = node.dynamic().downcast_ref::<VariableDeclarationNode>() {
                result.insert(&var.ident);
            }
            return Ok(());
        }).unwrap();
        return result;
    }

    fn generate_call_stmt<I>(&self, mut call: Box<FunctionCallNode>, function: &FunctionNode, result_name: Identifier, disjunct_names_provider: &mut I) -> Box<dyn StmtNode>
        where I: Iterator<Item = (Identifier, Identifier)>
    {
        let mut inlined_function_body = function.implementation.dynamic().downcast_ref::<ImplementedFunctionNode>().unwrap().body.clone();
        let mut inlined_function_variable_names = Self::find_variable_names(&*inlined_function_body);

        for param in &function.params {
            inlined_function_variable_names.insert(&param.ident);
        }

        let rename_variables: HashMap<Identifier, Identifier> = inlined_function_variable_names.iter()
            .map(|name| ((**name).clone(), disjunct_names_provider.next().unwrap().1))
            .collect();

        let call_pos = call.get_annotation().clone();
        let mut param_value_definitions: Vec<Box<dyn StmtNode>> = Vec::new();

        let parameter_count = function.params.len();
        let mut given_param_iter = call.params.drain(..);

        for i in 0..parameter_count {
            let given_param = given_param_iter.next().unwrap();
            let formal_param = &function.params[i];
            let param_value_definition = Box::new(VariableDeclarationNode::new(
                given_param.get_annotation().clone(), 
                rename_variables.get(&formal_param.ident).unwrap().clone(), 
                formal_param.param_type.dyn_clone(), 
                Some(given_param)));
            param_value_definitions.push(param_value_definition);
        }

        inlined_function_body.transform(&mut RenameTransformer {
            renaming: &rename_variables
        });
        inlined_function_body.transform(&mut TransformReturnTransformer {
            result_name: &result_name
        });

        param_value_definitions.push(inlined_function_body);

        let result = Box::new(BlockNode::new(call_pos, param_value_definitions));
        return result;
    }

    fn inline_calls_in_stmt<I>(&self, mut stmt: Box<dyn StmtNode>, disjunct_names_provider: &mut I) -> Vec<Box<dyn StmtNode>>
        where I: Iterator<Item = (Identifier, Identifier)>
    {
        let mut calls_to_inline = Vec::new();
        {
            stmt.transform(&mut InlineCallsInTopLevelExpressionsTransformer {
                disjunct_names_provider: disjunct_names_provider,
                inline_calls: &mut calls_to_inline
            });
        }
        let called_functions: Vec<&'a FunctionNode> = calls_to_inline.iter()
            .map(|(_temp_value_name, temp_value_call)| temp_value_call)
            .map(|temp_value_call| {
                self.lookup_function(&temp_value_call.function, &temp_value_call.get_annotation())
            })
            .collect::<Result<_, _>>().unwrap();

        debug_assert_eq!(called_functions.len(), calls_to_inline.len());

        let mut result_stmts: Vec<Box<dyn StmtNode>> = calls_to_inline.iter()
            .zip(called_functions.iter())
            .map(|((temp_value_name, temp_value_call), called_function)| {
                let result = Box::new(VariableDeclarationNode::new(
                    temp_value_call.get_annotation().clone(), 
                    temp_value_name.clone(), 
                    called_function.result.dyn_clone(), 
                    None)) as Box<dyn StmtNode>;
                return result;
            })
            .collect();
        
        result_stmts.extend(calls_to_inline.drain(..)
            .zip(called_functions.iter())
            .map(|((temp_value_name, temp_value_call), called_function)| {
                let result = self.generate_call_stmt(temp_value_call, *called_function, temp_value_name, disjunct_names_provider);
                return result;
            }));

        result_stmts.push(stmt);
        return result_stmts;
    }

    fn lookup_function(&self, name: &Identifier, pos: &TextPosition) -> Result<&'a FunctionNode, CompileError>
    {
        if let Some(def) = self.scope_stack.find_definition(name) {
            if def.is_global() {
                Ok(self.inlinable_functions.iter().find(|func| func.ident == *name).unwrap())
            } else {
                Err(CompileError::new(pos.clone(), format!("Function call to {} does not refer to a top level function", name), ErrorType::UndefinedSymbol))
            }
        } else {
            Err(CompileError::new(pos.clone(), format!("Cannot resolve symbol {}", name), ErrorType::UndefinedSymbol))
        }
    }

    fn inline_calls_in_block(&mut self, mut block: Box<BlockNode>) -> Box<BlockNode>
    {
        self.scope_stack.enter(&*block);
        let result_stmts = {
            let mut disjunct_names_provider = self.scope_stack.rename_disjunct((0..).map(|i| Identifier::auto(i)));
            block.stmts.drain(..).flat_map(|stmt| self.inline_calls_in_stmt(stmt, &mut disjunct_names_provider)).collect()
        };
        self.scope_stack.exit();
        let mut result = Box::new(BlockNode::new(block.get_annotation().clone(), result_stmts));
        result.transform(self);
        return result;
    }
}

impl<'a> Transformer for InlineCallsTransformer<'a>
{
    fn before(&mut self, node: &dyn Node)
    {
        if let Some(function) = node.dynamic().downcast_ref::<FunctionNode>() {
            self.scope_stack.enter(function);
        } else if let Some(block) = node.dynamic().downcast_ref::<BlockNode>() {
            self.scope_stack.enter(block);
        }
    }

    fn after(&mut self, node: &dyn Node)
    {
        if let Some(_function) = node.dynamic().downcast_ref::<FunctionNode>() {
            self.scope_stack.exit();
        } else if let Some(_block) = node.dynamic().downcast_ref::<BlockNode>() {
            self.scope_stack.exit();
        }
    }

    fn transform_stmt(&mut self, node: Box<dyn StmtNode>) -> Box<dyn StmtNode>
    {
        match cast::<dyn StmtNode, BlockNode>(node) {
            Ok(block) => self.inline_calls_in_block(block),
            Err(node) => match cast::<dyn StmtNode, IfNode>(node) {
                Ok(if_node) => Box::new(IfNode::new(if_node.get_annotation().clone(), if_node.condition, self.inline_calls_in_block(if_node.block))),
                Err(node) => match cast::<dyn StmtNode, WhileNode>(node) {
                    Ok(while_node) => Box::new(WhileNode::new(while_node.get_annotation().clone(), while_node.condition, self.inline_calls_in_block(while_node.block))),
                    Err(mut node) => {
                        node.transform(self);
                        node
                    }
                }
            }
        }
    }

    fn transform_expr(&mut self, node: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
    {
        node
    }
}

pub fn inline(program: &mut Program) 
{
    for i in 0..program.functions.len() {
        program.functions.swap(0, i);
        let (process_function, other_functions) = program.functions.split_first_mut().unwrap();
        let mut transformer = InlineCallsTransformer::new(&other_functions);
        transformer.scope_stack.enter(&*process_function);
        if let Some(implementation) = process_function.implementation.dynamic_mut().downcast_mut::<ImplementedFunctionNode>() {
            take_mut::take(&mut implementation.body, |block| transformer.inline_calls_in_block(block));
        }
    }
}

#[cfg(test)]
use super::super::parser::Parse;
#[cfg(test)]
use super::super::lexer::lexer::lex;
#[cfg(test)]
use super::test_rename::RenameAutoVars;

// #[test]
// fn test_generate_call_stmt() {
//     let function = FunctionNode::parse(lex("fn min(a: int, b: int,): int {
//             if a < b {
//                 return a;
//             }
//             return b;
//         }").expect_next(&Token::BOF).unwrap()).unwrap();
//     let expr = cast::<dyn UnaryExprNode, FunctionCallNode>(UnaryExprNode::parse(lex("min(0, (a + 1) * a, )").expect_next(&Token::BOF).unwrap()).unwrap()).unwrap();
//     let expected: Box<dyn StmtNode> = StmtNode::parse(lex("{
//             let auto_0: int = 0;
//             let b: int = (a + 1) * a;
//             {
//                 if auto_0 < b {
//                     result = auto_0;
//                 }
//                 result = b;
//             }
//         }").expect_next(&Token::BOF).unwrap()).unwrap();
//     let program = Program::new(TextPosition::create(0, 0), vec![function]);
//     let mut inline_transformer = InlineCallsTransformer::new(&program.functions);
//     inline_transformer.scope_stack.enter(&vec![ParameterNode::new(TextPosition::create(0, 0), Identifier::new("a"), ArrTypeNode::test_val(0))]);
//     let mut actual: Box<dyn StmtNode> = inline_transformer.generate_call_stmt(expr, &program.functions[0], Identifier::new("result"));
//     actual.rename_auto_vars();
//     assert_eq!(&expected, &actual, "{} != {}", expected, actual);
// }

#[test]
fn test_inline() {
    let program = Program::parse(&mut lex("
        fn min(a: int, b: int,): int {
            if a < b {
                return a;
            }
            return b;
        }
        
        fn max(a: int, b: int,): int {
            if b < a {
                return a;
            }
            return b;
        }

        fn clamp(a: int, lower: int, upper: int,): int {
            return min(max(a, lower, ), upper, );
        }")).unwrap();

    let expected = Program::parse(&mut lex("
        fn clamp(a: int, lower: int, upper: int, ): int {
            let auto_0: int;
            {
                    let auto_3: int;
                    {
                            let auto_4: int = a;
                            let auto_5: int = lower;
                            {
                                    if auto_5 < auto_4 {
                                            auto_3 = auto_4;
                                    }
                                    auto_3 = auto_5;
                            }
                    }
                    let auto_1: int = auto_3;
                    let auto_2: int = upper;
                    {
                            if auto_1 < auto_2 {
                                    auto_0 = auto_1;
                            }
                            auto_0 = auto_2;
                    }
            }
            return auto_0;
        }

        fn min(a: int, b: int, ): int {
                if a < b {
                        return a;
                }
                return b;
        }

        fn max(a: int, b: int, ): int {
                if b < a {
                        return a;
                }
                return b;
        }")).unwrap();
    
    let mut actual = program;
    inline(&mut actual);
    actual.rename_auto_vars();
    
    assert_eq!(&expected, &actual, "{} != {}", expected, actual);
}