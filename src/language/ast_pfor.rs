use super::super::analysis::types::get_expression_type_nonvoid;
use super::position::TextPosition;
use super::types::*;
use super::error::{CompileError, ErrorType};
use super::identifier::{Name, Identifier, BuiltInIdentifier};
use super::ast::*;
use super::ast_expr::*;
use super::ast_statement::*;
use super::gwaihir_writer::*;
use super::scopes::*;
use super::symbol::*;
use super::concrete_views::*;
use feanor_la::la::mat::*;
use feanor_la::la::matrix_row_col::*;
use feanor_la::algebra::rat::*;

use std::collections::HashMap;

impl CompileError {

    fn no_affine_expression(e: &Expression) -> CompileError {
        CompileError::new(
            e.pos(), 
            format!("Expressions used as indices in parallel for loops must be affine linear, got `{}`", DisplayWrapper::from(e)), 
            ErrorType::IllegalPForIndexExpression
        )
    }
}

#[derive(Debug, Clone)]
pub struct ArrayEntryAccessData {
    pos: TextPosition,
    alias: Name,
    indices: Vec<Expression>,
    writeable: bool
}

impl PartialEq for ArrayEntryAccessData {

    fn eq(&self, rhs: &ArrayEntryAccessData) -> bool {
        self.alias == rhs.alias && self.indices == rhs.indices && self.writeable == rhs.writeable
    }
}

impl ArrayEntryAccessData {

    pub fn new(pos: TextPosition, alias: Name, indices: Vec<Expression>, writeable: bool) -> Self {
        ArrayEntryAccessData {
            pos, alias, indices, writeable
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArrayAccessPatternData {
    pos: TextPosition,
    pub array: Expression,
    pub accesses: Vec<ArrayEntryAccessData>
}

impl ArrayAccessPatternData {

    pub fn new(pos: TextPosition, array: Expression, accesses: Vec<ArrayEntryAccessData>) -> Self {
        ArrayAccessPatternData {
            pos, array, accesses
        }
    }
}

impl PartialEq for ArrayAccessPatternData {

    fn eq(&self, rhs: &ArrayAccessPatternData) -> bool {
        self.array == rhs.array && self.accesses == rhs.accesses
    }
}

#[derive(Debug, Clone)]
pub struct AffineTransform1D {
    linear_part: Vector<VectorOwned<r64>, r64>,
    affine_part: r64
}

impl AffineTransform1D {

    pub fn zero(n: usize) -> AffineTransform1D {
        Self::only_affine(n, r64::ZERO)
    }

    pub fn only_affine(n: usize, additive_part: r64) -> AffineTransform1D {
        AffineTransform1D {
            linear_part: Vector::zero(n).to_owned(),
            affine_part: additive_part
        }
    }

    pub fn scale(&mut self, factor: r64) {
        self.affine_part *= factor;
        self.linear_part *= factor;
    }

    pub fn add(mut self, rhs: AffineTransform1D) -> AffineTransform1D {
        self.linear_part += rhs.linear_part;
        self.affine_part += rhs.affine_part;
        return self;
    }

    pub fn neg(mut self) -> AffineTransform1D {
        self.linear_part *= -r64::ONE;
        self.affine_part = -self.affine_part;
        return self;
    }
}

fn to_constant(expr: &Expression) -> Result<r64, ()> {
    fn apply<'a, I: Iterator<Item = &'a Expression>, F: FnMut(r64, r64) -> r64>(iter: I, mut f: F) -> Result<r64, ()> {
        iter.map(|p| to_constant(p))
            .fold(Ok(r64::ZERO), |a, b| a.and_then(|v| b.map(|u| f(u, v))))
    }
    match expr {
        Expression::Call(call) if call.function == BuiltInIdentifier::FunctionAdd => {
            apply(call.parameters.iter(), |x, y| x + y)
        },
        Expression::Call(call) if call.function == BuiltInIdentifier::FunctionMul => {
            apply(call.parameters.iter(), |x, y| x * y)
        },
        Expression::Call(call) if call.function == BuiltInIdentifier::FunctionUnaryNeg => {
            assert!(call.parameters.len() == 1);
            Ok(-to_constant(&call.parameters[0])?)
        },
        Expression::Call(call) if call.function == BuiltInIdentifier::FunctionUnaryDiv => {
            assert!(call.parameters.len() == 1);
            Ok(r64::ONE / to_constant(&call.parameters[0])?)
        },
        Expression::Literal(lit) => Ok(r64::from(&lit.value)),
        _ => Err(())
    }
}

fn to_affine_transform(expr: &Expression, vars: &HashMap<Identifier, usize>) -> Result<AffineTransform1D, ()> {
    
    match expr {
        Expression::Call(call) if call.function == BuiltInIdentifier::FunctionMul => {
            let mut factor = r64::ONE;
            let mut transform: Option<AffineTransform1D> = None;
            for p in &call.parameters {
                if let Ok(constant) = to_constant(p) {
                    factor *= constant;
                } else if transform.is_some() {
                    return Err(());
                } else {
                    transform = Some(to_affine_transform(p, vars)?);
                }
            }
            let mut result = transform.unwrap_or_else(|| AffineTransform1D::only_affine(vars.len(), r64::ONE));
            result.scale(factor);
            return Ok(result);
        },
        Expression::Call(call) if call.function == BuiltInIdentifier::FunctionAdd => {
            let mut result = to_affine_transform(&call.parameters[0], vars)?;
            for p in call.parameters.iter().skip(1) {
                result = result.add(to_affine_transform(p, vars)?);
            }
            return Ok(result);
        },
        Expression::Call(call) if call.function == BuiltInIdentifier::FunctionUnaryNeg => {
            assert!(call.parameters.len() == 1);
            let result = to_affine_transform(&call.parameters[0], vars)?;
            return Ok(result.neg());
        },
        Expression::Call(call) if call.function == BuiltInIdentifier::FunctionUnaryDiv => {
            assert!(call.parameters.len() == 1);
            let result = to_constant(&call.parameters[0])?;
            return Ok(AffineTransform1D::only_affine(vars.len(), r64::ONE / result));
        },
        Expression::Variable(var) => {
            let mut result = AffineTransform1D::zero(vars.len());
            *result.linear_part.at_mut(*vars.get(&var.identifier).unwrap()) = r64::ONE;
            return Ok(result);
        },
        Expression::Literal(lit) => {
            return Ok(AffineTransform1D::only_affine(vars.len(), r64::from(&lit.value)));
        },
        _ => {
            return Err(());
        }
    };
}

#[derive(Debug, PartialEq, Clone)]
pub struct AffineTransform {
    pub linear_part: Matrix<MatrixOwned<r64>, r64>,
    pub affine_part: Vector<VectorOwned<r64>, r64>
}

impl AffineTransform {

    pub fn components<'a>(&'a self) -> impl 'a + Iterator<Item = (Vector<MatrixRow<r64, MatrixRef<'a, MatrixOwned<r64>, r64>>, r64>, r64)> {
        (0..self.linear_part.row_count()).map(move |i| (self.linear_part.row(i), self.affine_part[i]))
    }
}

#[derive(Debug)]
pub struct ArrayEntryAccess {
    pos: TextPosition,
    alias: Name,
    pub writeable: bool,
    entry: AffineTransform,
    ty: Option<Type>
}

impl AstNodeFuncs for ArrayEntryAccess {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for ArrayEntryAccess {}

impl PartialEq for ArrayEntryAccess {

    fn eq(&self, rhs: &ArrayEntryAccess) -> bool {
        self.alias == rhs.alias && self.writeable == rhs.writeable && self.entry == rhs.entry
    }
}

impl ArrayEntryAccess {

    fn from_data(
        pos: TextPosition, 
        alias: Name, 
        writeable: bool, 
        indices: Vec<Expression>, 
        vars: &HashMap<Identifier, usize>
    ) -> Result<Self, CompileError> {
        let mut transform_linear_part = Matrix::zero(indices.len(), vars.len()).to_owned();
        let mut transform_affine_part = Vector::zero(indices.len()).to_owned();
        for i in 0..indices.len() {
            let transform = to_affine_transform(&indices[i], vars)
                .map_err(|_| CompileError::no_affine_expression(&indices[i]))?;
            *&mut transform_linear_part.submatrix_mut(i..=i, ..) += Matrix::row_vec(transform.linear_part);
            *transform_affine_part.at_mut(i) += transform.affine_part;
        }
        let entry = AffineTransform {
            linear_part: transform_linear_part,
            affine_part: transform_affine_part
        };
        let ty = None;
        return Ok(ArrayEntryAccess {
            pos, alias, writeable, entry, ty
        });
    }

    pub fn transform(&self) -> &AffineTransform {
        &self.entry
    }

    fn store_type(&mut self, array_type: &Type) {
        let ty = match array_type {
            Type::Static(s) => s.get_base().as_scalar_type(self.writeable).with_concrete_view(VIEW_INDEX),
            Type::View(v) => v.view_onto.get_base()
                .as_scalar_type(self.writeable)
                .with_concrete_view_dyn(ViewComposed::compose(VIEW_INDEX, v.get_concrete().dyn_clone())),
            Type::Function(_) => panic!("function type cannot be used in pfor")
        };
        self.ty = Some(ty);
    }
}

impl SymbolDefinitionFuncs for ArrayEntryAccess {
    
    fn get_name(&self) -> &Name {
        &self.alias
    }

    fn get_name_mut(&mut self) -> &mut Name {
        &mut self.alias
    }

    fn cast_statement_mut(&mut self) -> Option<&mut dyn Statement> {
        None
    }

    fn get_type(&self) -> &Type {
        self.ty.as_ref().unwrap()
    }
}

impl SymbolDefinition for ArrayEntryAccess {}

#[derive(Debug)]
pub struct ArrayAccessPattern {
    pos: TextPosition,
    pub array: Expression,
    pub pattern: Vec<ArrayEntryAccess>
}

impl PartialEq for ArrayAccessPattern {

    fn eq(&self, rhs: &ArrayAccessPattern) -> bool {
        self.array == rhs.array && self.pattern == rhs.pattern
    }
}

impl AstNodeFuncs for ArrayAccessPattern {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for ArrayAccessPattern {}

impl ArrayAccessPattern {

    fn from_data(
        pos: TextPosition, 
        array: Expression, 
        accesses: Vec<ArrayEntryAccessData>, 
        vars: &HashMap<Identifier, usize>
    ) -> Result<Self, CompileError> {
        Ok(ArrayAccessPattern {
            pos: pos,
            array: array,
            pattern: accesses.into_iter()
                .map(|a| ArrayEntryAccess::from_data(a.pos, a.alias, a.writeable, a.indices, vars))
                .collect::<Result<Vec<ArrayEntryAccess>, CompileError>>()?
        })
    }

    pub fn accessed_entries(&self) -> impl Iterator<Item = &ArrayEntryAccess> {
        self.pattern.iter()
    }

    pub fn store_array_entry_types<'a, D>(&mut self, scopes: &D) -> Result<(), CompileError> 
        where D: DefinitionEnvironment<'a, 'a>
    {
        for entry in &mut self.pattern {
            entry.store_type(get_expression_type_nonvoid(&self.array, scopes)?);
        }
        return Ok(());
    }
}

#[derive(Debug)]
pub struct ParallelFor {
    pos: TextPosition,
    pub index_variables: Vec<Declaration>,
    pub used_variables: Vec<Expression>,
    pub access_pattern: Result<Vec<ArrayAccessPattern>, CompileError>,
    pub body: Block
}

impl PartialEq for ParallelFor {

    fn eq(&self, rhs: &ParallelFor) -> bool {
        self.index_variables == rhs.index_variables && 
            self.access_pattern.as_ref().map_err(|_| ()) == rhs.access_pattern.as_ref().map_err(|_| ()) && 
            self.body == rhs.body
    }
}

impl ParallelFor {

    fn calc_access_pattern(patterns: Vec<ArrayAccessPatternData>, vars: &HashMap<Identifier, usize>) -> Result<Vec<ArrayAccessPattern>, CompileError> {
        let mut result = Vec::new();
        for p in patterns.into_iter() {
            result.push(ArrayAccessPattern::from_data(p.pos, p.array, p.accesses, vars)?);
        }
        return Ok(result);
    }
    
    pub fn new(
        pos: TextPosition, 
        index_variables: Vec<(Name, Type)>, 
        patterns: Vec<ArrayAccessPatternData>, 
        body: Block
    ) -> Self {

        let mut vars = HashMap::new();
        let mut index_vars = Vec::new();
        for (name, ty) in index_variables.into_iter() {
            vars.insert(Identifier::Name(name.clone()), index_vars.len());
            index_vars.push(Declaration {
                pos: pos.clone(),
                name: name,
                var_type: ty
            });
        }
        let mut used_variables = Vec::new();
        for pattern in &patterns {
            for access in &pattern.accesses {
                for index in &access.indices {
                    index.traverse_preorder(&mut |e| {
                        if let Expression::Variable(v) = e {
                            if v.identifier.is_name() {
                                if vars.get(&v.identifier).is_none() {
                                    vars.insert(v.identifier.clone(), used_variables.len() + index_vars.len());
                                    used_variables.push(e.clone());
                                }
                            }
                        }
                        return RECURSE;
                    }).unwrap();
                }
            }
        }

        return ParallelFor {
            pos: pos,
            index_variables: index_vars,
            used_variables: used_variables,
            access_pattern: Self::calc_access_pattern(patterns, &vars),
            body: body
        }
    }

    pub fn access_pattern(&self) -> Result<impl Iterator<Item = &ArrayAccessPattern>, CompileError> {
        self.access_pattern.as_ref().map(|v| v.iter()).map_err(|e| e.clone())
    }

    pub fn access_pattern_mut(&mut self) -> Result<impl Iterator<Item = &mut ArrayAccessPattern>, CompileError> {
        self.access_pattern.as_mut().map(|v| v.iter_mut()).map_err(|e| e.clone())
    }

    pub fn variable_by_index(&self, i: usize, pos: &TextPosition) -> Expression {
        if i < self.index_variables.len() {
            Expression::Variable(Variable {
                pos: pos.clone(),
                identifier: Identifier::Name(self.index_variables[i].get_name().clone())
            })
        } else {
            self.used_variables[i - self.index_variables.len()].clone()
        }
    }
}

impl AstNodeFuncs for ParallelFor {

    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl AstNode for ParallelFor {}

impl StatementFuncs for ParallelFor {
        
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(&self.body))
    }

    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(&mut self.body))
    }

    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(
            self.access_pattern.iter().flat_map(|p| p.iter()).map(|p| &p.array)
                .chain(self.used_variables.iter())
        )
    }

    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(
            self.access_pattern.iter_mut().flat_map(|p| p.iter_mut()).map(|p| &mut p.array)
                .chain(self.used_variables.iter_mut())
        )
    }

    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(
            self.index_variables.iter().map(|d| d.get_name())
                .chain(
                    self.access_pattern.iter().flat_map(|p| p.iter()).map(|p| &p.array)
                        .chain(self.used_variables.iter())
                        .flat_map(|e| e.names())
                )
        )
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(
            self.index_variables.iter_mut().map(|d| d.get_name_mut())
            .chain(
                self.access_pattern.iter_mut().flat_map(|p| p.iter_mut()).map(|p| &mut p.array)
                    .chain(self.used_variables.iter_mut())
                    .flat_map(|e| e.names_mut())
            )
        )
    }

    fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStackConst<'_, 'a>, 
        f: &mut dyn FnMut(&'a dyn Statement, &DefinitionScopeStackConst<'_, 'a>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        let recurse = f(self, parent_scopes);
        let mut scopes = parent_scopes.child_stack();
        for def in &self.index_variables {
            scopes.register(def.get_name().clone(), def as &dyn SymbolDefinition);
        }
        for array in self.access_pattern.as_ref()?.iter() {
            for access in array.pattern.iter() {
                scopes.register_symbol(access);
            }
        }

        return self.body.traverse_preorder_base(&scopes, f, recurse);
    }

    fn traverse_preorder_mut<'a>(
        &'a mut self, 
        parent_scopes: &DefinitionScopeStackMut<'_, '_>, 
        f: &mut dyn FnMut(&mut dyn Statement, &DefinitionScopeStackMut<'_, '_>) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        let recurse = f(self, parent_scopes);
        let mut scopes = parent_scopes.child_stack();
        for def in &mut self.index_variables {
            scopes.register_symbol(def as &mut dyn SymbolDefinition);
        }
        for array in self.access_pattern.as_mut()?.iter_mut() {
            for access in array.pattern.iter_mut() {
                scopes.register_symbol(access);
            }
        }

        return self.body.traverse_preorder_mut_base(&scopes, f, recurse);
    }
}

impl Statement for ParallelFor {}

#[cfg(test)]
use super::super::language::test::*;

#[test]
fn test_symbols_pfor() {
    let pfor = ParallelFor::parse(&mut fragment_lex("

        pfor a: int, b: int, with write this[a, 2 * b,] as even, write this[a, 2 * b + 1,] as odd, in array with this[a, i,] as foo, in array2 {
            even = foo + a;
            odd = even - b;
        }

    "), &mut ParserContext::new()).unwrap();
    
    let mut scopes = DefinitionScopeStackConst::new();
    let defs = [testdef("array"), testdef("array2"), testdef("i")];
    for d in &defs {
        scopes.register_symbol(d);
    }

    let mut counter = 0;
    pfor.traverse_preorder(&scopes, &mut |_, scopes| {
        counter += 1;
        if counter > 1 {
            assert!(scopes.get(&Name::l("a")).is_some());
            assert!(scopes.get(&Name::l("b")).is_some());
            assert!(scopes.get(&Name::l("i")).is_some());
            assert!(scopes.get(&Name::l("even")).is_some());
            assert!(scopes.get(&Name::l("odd")).is_some());
            assert!(scopes.get(&Name::l("foo")).is_some());
        }
        return RECURSE;
    }).unwrap();

    assert_eq!(3, counter);
}

#[test]
fn test_transform_pfor() {
    let pfor = ParallelFor::parse(&mut fragment_lex("

        pfor a: int, b: int, with write this[a, 2 * b,] as even, write this[a, 2 * b + 1,] as odd, in array with this[a, i,] as foo, in array2 {
            even = foo + a;
            odd = even - b;
        }

    "), &mut ParserContext::new()).unwrap();
    
    assert_eq!(1, pfor.used_variables.len());
    assert_eq!(2, pfor.index_variables.len());

    let t1 = pfor.access_pattern().unwrap().next().unwrap().accessed_entries().next().unwrap().transform();

    #[rustfmt::skip]
    assert_eq!(Matrix::from_array([[r64::ONE,  r64::ZERO,    r64::ZERO], 
                                   [r64::ZERO, r64::from(2), r64::ZERO]]), t1.linear_part);
    assert_eq!(Vector::from_array([r64::ZERO, r64::ZERO]), t1.affine_part);
    
    let t2 = pfor.access_pattern().unwrap().next().unwrap().accessed_entries().skip(1).next().unwrap().transform();

    #[rustfmt::skip]
    assert_eq!(Matrix::from_array([[r64::ONE,  r64::ZERO,    r64::ZERO], 
                                   [r64::ZERO, r64::from(2), r64::ZERO]]), t2.linear_part);
    assert_eq!(Vector::from_array([r64::ZERO, r64::ONE]), t2.affine_part);
}