use super::super::language::prelude::*;
use super::super::analysis::symbol::*;
use super::super::analysis::scope::*;
use super::super::util::ref_eq::*;
use super::super::util::dynamic::Dynamic;
use super::super::language::backend::OutputError;
use super::CudaContext;

use std::borrow::Borrow;
use std::hash::Hash;
use std::collections::HashSet;
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum TargetLanguageFunction<'a> {
    Kernel(Ref<'a, ParallelFor>), Function(Ref<'a, Function>)
}

impl<'a> TargetLanguageFunction<'a> {
    pub fn unwrap_kernel(&self) -> Ref<'a, ParallelFor> {
        match self {
            TargetLanguageFunction::Kernel(ker) => *ker,
            TargetLanguageFunction::Function(_) => panic!("Called unwrap_kernel() on a function reference")
        }
    }

    pub fn unwrap_function(&self) -> Ref<'a, Function> {
        match self {
            TargetLanguageFunction::Kernel(_) => panic!("Called unwrap_function() on a kernel reference"),
            TargetLanguageFunction::Function(func) => *func
        }
    }
}

pub struct KernelInfo<'a> {
    pub pfor: &'a ParallelFor,
    pub used_variables: HashSet<Ref<'a, dyn SymbolDefinition>>,
    pub called_from: TargetLanguageFunction<'a>,
    pub kernel_name: u32
}

pub struct FunctionInfo<'a> {
    pub function: &'a Function,
    pub called_from: HashSet<TargetLanguageFunction<'a>>
}

#[allow(unused)]
fn collect_functions<'stack, 'ast: 'stack>(program: &'ast Program, context: &mut dyn CudaContext<'stack, 'ast>) -> Result<(HashMap<Ref<'ast, Function>, FunctionInfo<'ast>>, HashMap<Ref<'ast, ParallelFor>, KernelInfo<'ast>>), OutputError> {
    let mut functions: HashMap<Ref<'ast, Function>, FunctionInfo<'ast>> = HashMap::new();
    let mut kernels: HashMap<Ref<'ast, ParallelFor>, KernelInfo<'ast>> = HashMap::new();
    for item in &program.items {
        functions.insert(Ref::from(&**item), FunctionInfo {
            function: item,
            called_from: HashSet::new()
        });
    }
    let scopes: ScopeStack<&'ast dyn SymbolDefinition> = ScopeStack::new(&program.items);
    for item in &program.items {
        if let Some(body) = &item.body {
            collect_calls_and_kernels(body, TargetLanguageFunction::Function(Ref::from(&**item)), &scopes.child_scope(&**item), &mut functions, &mut kernels, context)?;
        }
    }
    return Ok((functions, kernels));
}

fn collect_calls_and_kernels<'b, 'stack, 'ast: 'stack>(block: &'ast Block, 
    parent: TargetLanguageFunction<'ast>, 
    scopes: &ScopeStack<'b, &'ast dyn SymbolDefinition>, 
    functions: &mut HashMap<Ref<'ast, Function>, FunctionInfo<'ast>>, 
    kernels: &mut HashMap<Ref<'ast, ParallelFor>, KernelInfo<'ast>>, 
    context: &mut dyn CudaContext<'stack, 'ast>) -> Result<(), OutputError>
{
    let this_scopes = scopes.child_scope(block);
    for statement in &block.statements {
        for expr in statement.iter() {
            set_called_from(expr, parent, functions, &this_scopes)?;
        }
        if let Some(pfor) = (**statement).dynamic().downcast_ref::<ParallelFor>() {
            let mut kernel = KernelInfo {
                pfor: pfor, 
                called_from: parent,
                used_variables: HashSet::new(),
                kernel_name: context.generate_unique_identifier()
            };
            pfor.body.scan_top_level_expressions(&mut |e| add_variable_uses(e, &mut kernel, &this_scopes));
            kernels.insert(Ref::from(pfor), kernel);
            collect_calls_and_kernels(&pfor.body, TargetLanguageFunction::Kernel(Ref::from(pfor)), &this_scopes.child_scope(pfor), functions, kernels, context)?;
        } else {
            for child_block in statement.iter() {
                collect_calls_and_kernels(child_block, parent, &this_scopes, functions, kernels, context)?;
            }
        }
    }
    return Ok(());
}

fn set_called_from<'a, 'b>(expr: &'a Expression,
    parent: TargetLanguageFunction<'a>,
    functions: &mut HashMap<Ref<'a, Function>, FunctionInfo<'a>>,
    scopes: &ScopeStack<'b, &'a dyn SymbolDefinition>) -> Result<(), OutputError>
{
    match expr {
        Expression::Call(call) => {
            if let Expression::Variable(var) = &call.function {
                if let Identifier::Name(name) = &var.identifier {
                    let definition = *scopes.get(&name).expect("Unresolved symbol");
                    if let Some(function) = definition.dynamic().downcast_ref::<Function>() {
                        functions.get_mut(&RefEq::from(function)).unwrap().called_from.insert(parent);
                    } else {
                        panic!("Not a function");
                    }
                }
            } else {
                return Err(OutputError::UnsupportedCode(call.pos().clone(), format!("Cannot call dynamic expression")))
            }
            for param in &call.parameters {
                set_called_from(param, parent, functions, scopes)?;
            }
        },
        Expression::Literal(_) => {},
        Expression::Variable(_) => {}
    };
    return Ok(());
}

fn add_variable_uses<'a, 'b>(expr: &'a Expression, 
    parent: &mut KernelInfo<'a>, 
    out_of_kernel_variables: &ScopeStack<'b, &'a dyn SymbolDefinition>) 
{
    match expr {
        Expression::Call(call) => {
            add_variable_uses(&call.function, parent, out_of_kernel_variables);
            for param in &call.parameters {
                add_variable_uses(param, parent, out_of_kernel_variables);
            }
        },
        Expression::Literal(_) => {},
        Expression::Variable(var) => if let Identifier::Name(name) = &var.identifier {
            if let Some(definition) = out_of_kernel_variables.non_global_definitions().find(|def| def.0 == name) {
                parent.used_variables.insert(Ref::from(*definition.1));
            } else {
                // this symbol is defined in the kernel, so no param is required
            }
        }
    }
}

#[cfg(test)]
use super::super::lexer::lexer::lex;
#[cfg(test)]
use super::super::parser::Parser;
#[cfg(test)]
use super::context::CudaContextImpl;

#[test]
fn test_collect_functions() {
    let program = Program::parse(&mut lex("
        fn foo(a: &int[,],): int {
            pfor i: int, with this[i,], in a {
                a[i,] = bar(a[i,],);
            }
            return bar(1,);
        }
        fn bar(a: int,): int {
            return a + 1;
        }
    ")).unwrap();
    let mut counter: u32 = 0;
    let mut context = CudaContextImpl::new(&[], &mut counter);
    let (mut functions, kernels) = collect_functions(&program, &mut context).unwrap();

    assert_eq!(2, functions.len());
    assert_eq!(1, kernels.len());

    let kernel = kernels.iter().next().unwrap().1;
    let kernel_caller = functions.remove::<RefEq<_>>(kernel.called_from.unwrap_function().borrow()).unwrap();
    let kernel_callee = functions.into_iter().next().unwrap().1;

    let mut expected_callers = HashSet::new();
    expected_callers.insert(TargetLanguageFunction::Function(Ref::from(kernel_caller.function)));
    expected_callers.insert(TargetLanguageFunction::Kernel(Ref::from(kernel.pfor)));
    assert_eq!(expected_callers, kernel_callee.called_from);
    assert_eq!(Ref::from(kernel_caller.function), kernel.called_from.unwrap_function());
    assert_eq!(1, kernel.used_variables.len());
    assert_eq!(Name::l("a"), *kernel.used_variables.iter().next().unwrap().get_name());
}