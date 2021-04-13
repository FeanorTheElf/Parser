use super::super::language::prelude::*;
use super::super::language::ast_pfor::*;
use super::sort_by_name::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum TargetLanguageFunction<'a> {
    Kernel(Ptr<'a, ParallelFor>),
    Function(Ptr<'a, Function>),
}

impl<'a> TargetLanguageFunction<'a> {

    pub fn unwrap_kernel(&self) -> Ptr<'a, ParallelFor> {
        match self {
            TargetLanguageFunction::Kernel(ker) => *ker,
            TargetLanguageFunction::Function(_) => {
                panic!("Called unwrap_kernel() on a function reference")
            }
        }
    }

    pub fn unwrap_function(&self) -> Ptr<'a, Function> {
        match self {
            TargetLanguageFunction::Kernel(_) => {
                panic!("Called unwrap_function() on a kernel reference")
            }
            TargetLanguageFunction::Function(func) => *func,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            TargetLanguageFunction::Kernel(_) => false,
            TargetLanguageFunction::Function(_) => true,
        }
    }

    pub fn is_kernel(&self) -> bool {
        match self {
            TargetLanguageFunction::Kernel(_) => true,
            TargetLanguageFunction::Function(_) => false,
        }
    }
}

pub type SortByNameSymbolDefinition<'a> =
    super::super::util::cmp::Comparing<&'a dyn SymbolDefinition, SortByName>;

pub struct KernelInfo<'a> {
    pub pfor: &'a ParallelFor,
    pub used_variables: BTreeSet<SortByNameSymbolDefinition<'a>>,
    pub called_from: TargetLanguageFunction<'a>
}

pub struct FunctionInfo<'a> {
    pub function: &'a Function,
    pub called_from: HashSet<TargetLanguageFunction<'a>>,
    pub called_from_device: bool,
    pub called_from_host: bool,
}

pub type OutputError = !;

pub fn collect_functions<'a, 'ast>(
    program: &'ast Program
) -> Result<
    (
        HashMap<Ptr<'ast, Function>, FunctionInfo<'ast>>,
        HashMap<Ptr<'ast, ParallelFor>, KernelInfo<'ast>>,
    ),
    OutputError,
> {
    let mut functions: HashMap<Ptr<'ast, Function>, FunctionInfo<'ast>> = HashMap::new();
    let mut kernels: HashMap<Ptr<'ast, ParallelFor>, KernelInfo<'ast>> = HashMap::new();

    program.for_functions(&mut |f, _| {
        functions.insert(
            Ptr::from(f),
            FunctionInfo {
                function: f,
                called_from: HashSet::new(),
                called_from_device: false,
                called_from_host: false,
            },
        );
        return Ok(());
    }).unwrap();

    program.for_functions(&mut |func, scopes| {
        let parent = TargetLanguageFunction::Function(Ptr::from(func));
        func.traverse_preorder(scopes, &mut |statement, scopes, _| 
            collect_calls_and_kernels(statement, parent, scopes, &mut functions, &mut kernels)
        )
    }).unwrap();

    return Ok((functions, kernels));
}

fn collect_calls_and_kernels<'a, 'b, 'ast>(
    statement: &'ast dyn Statement,
    parent: TargetLanguageFunction<'ast>,
    scopes: &DefinitionScopeStackConst<'b, 'ast>,
    functions: &mut HashMap<Ptr<'ast, Function>, FunctionInfo<'ast>>,
    kernels: &mut HashMap<Ptr<'ast, ParallelFor>, KernelInfo<'ast>>
) -> TraversePreorderResult {

    for expr in statement.expressions() {
        set_called_from(expr, parent, functions, scopes).unwrap();
    }
    if let Some(pfor) = statement.downcast::<ParallelFor>() {

        let mut kernel = KernelInfo {
            pfor: pfor,
            called_from: parent,
            used_variables: BTreeSet::new()
        };
        // collect all variables that are used inside the pfor but defined outside,
        // as these must be passed explicitly to the cuda kernel 
        pfor.traverse_preorder(scopes, &mut |statement, _| {
            for name in statement.names() {
                if let Some(used_var) = scopes.get(name) {
                    if scopes.get_global_scope().get(name).is_none() {
                        kernel.used_variables.insert(SortByNameSymbolDefinition::from(*used_var));
                    }
                }
            }
            return RECURSE;
        })?;
        kernels.insert(Ptr::from(pfor), kernel);

        let child_parent = TargetLanguageFunction::Kernel(Ptr::from(pfor));
        pfor.body.traverse_preorder(scopes, &mut |child_statement, child_scopes| 
            collect_calls_and_kernels(child_statement, child_parent, child_scopes, functions, kernels)
        )?;

        return DONT_RECURSE;
    } else {
        return RECURSE;
    }
}

fn set_called_from<'a, 'b>(
    expr: &'a Expression,
    parent: TargetLanguageFunction<'a>,
    functions: &mut HashMap<Ptr<'a, Function>, FunctionInfo<'a>>,
    scopes: &DefinitionScopeStackConst,
) -> Result<(), OutputError> {

    expr.traverse_preorder(&mut |expr: &Expression| {
        if let Expression::Call(call) = expr {
            if let Expression::Variable(var) = &call.function {
                if var.identifier.is_name() {
                    let function = scopes
                        .get_defined(var.identifier.unwrap_name(), call.pos())
                        .internal_error()
                        .downcast::<Function>()
                        .unwrap();
                    let function_info = functions.get_mut(&RefEq::from(function)).unwrap();
                    function_info.called_from.insert(parent);
                }
            } else {
                panic!();
            }
        }
        return RECURSE;
    }).unwrap();

    return Ok(());
}

#[cfg(test)]
use super::super::lexer::lexer::lex_str;
#[cfg(test)]
use super::super::parser::TopLevelParser;

#[test]

fn test_collect_functions() {

    let program = Program::parse(&mut lex_str("
        fn foo(a: &int[,],): int {
            pfor i: int, with this[i,] as entry, in a {
                entry = bar(entry,);
            }
            return bar(1,);
        }
        fn bar(a: int,): int {
            return a + 1;
        }
    "))
    .unwrap();

    let (mut functions, kernels) = collect_functions(&program).unwrap();

    assert_eq!(2, functions.len());
    assert_eq!(1, kernels.len());

    let kernel = kernels.iter().next().unwrap().1;

    let kernel_caller = functions
        .remove::<RefEq<_>>(&RefEq::from(&*kernel.called_from.unwrap_function()))
        .unwrap();

    let kernel_callee = functions.into_iter().next().unwrap().1;

    let mut expected_callers = HashSet::new();

    expected_callers.insert(TargetLanguageFunction::Function(Ptr::from(
        kernel_caller.function,
    )));
    expected_callers.insert(TargetLanguageFunction::Kernel(Ptr::from(kernel.pfor)));

    assert_eq!(expected_callers, kernel_callee.called_from);
    assert_eq!(
        Ptr::from(kernel_caller.function),
        kernel.called_from.unwrap_function()
    );
    assert_eq!(1, kernel.used_variables.len());
    assert_eq!(
        Name::l("a"),
        *kernel.used_variables.iter().next().unwrap().get_name()
    );
}
