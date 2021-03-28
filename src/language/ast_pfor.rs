use super::position::TextPosition;
use super::types::*;
use super::error::{CompileError, ErrorType};
use super::identifier::{Name, Identifier, BuiltInIdentifier};
use super::ast::*;
use super::ast_expr::*;
use super::ast_statement::*;
use feanor_la::la::mat::*;
use feanor_la::algebra::rat::*;

use std::collections::HashMap;

impl CompileError {

    fn no_affine_expression(e: &Expression) -> CompileError {
        CompileError::new(
            e.pos(), 
            format!("Expressions used as indices in parallel for loops must be affine linear, got {:?}", e), 
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
            linear_part: Vector::zero(n),
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
        self.linear_part.scale(-r64::ONE);
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
                } else if let Ok(t) = to_affine_transform(p, vars) {
                    transform = Some(t);
                } else {
                    return Err(());
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
            return Ok(AffineTransform1D::only_affine(vars.len(), r64::from(lit.value as i64)));
        },
        _ => {
            return Err(());
        }
    };
}

#[derive(Debug, PartialEq, Clone)]
pub struct AffineTransform {
    linear_part: Matrix<MatrixOwned<r64>, r64>,
    affine_part: Vector<VectorOwned<r64>, r64>
}

#[derive(Debug)]
pub struct ArrayEntryAccess {
    pos: TextPosition,
    alias: Name,
    writeable: bool,
    entry: AffineTransform
}

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
        let mut transform_linear_part = Matrix::zero(indices.len(), vars.len());
        let mut transform_affine_part = Vector::zero(indices.len());
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
        return Ok(ArrayEntryAccess {
            pos, alias, writeable, entry
        });
    }
}

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
                            if vars.get(&v.identifier).is_none() {
                                vars.insert(v.identifier.clone(), used_variables.len() + index_vars.len());
                                used_variables.push(e.clone());
                            }
                        }
                        return Ok(());
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
        unimplemented!()
    }

    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        unimplemented!()
    }

    fn traverse_preorder<'a>(
        &'a self, 
        parent_scopes: &DefinitionScopeStack, 
        f: &mut dyn FnMut(&'a Block, &DefinitionScopeStack) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        let mut scopes = parent_scopes.child_stack();
        for def in &self.index_variables {
            scopes.register(def.name.clone(), def as &dyn SymbolDefinition);
        }
        self.body.traverse_preorder(&scopes, f)
    }

    fn traverse_preorder_mut(
        &mut self, 
        parent_scopes: &DefinitionScopeStackMut, 
        f: &mut dyn FnMut(&mut Block, &DefinitionScopeStackMut) -> TraversePreorderResult
    ) -> Result<(), CompileError> {
        let mut scopes = parent_scopes.child_stack();
        for def in &mut self.index_variables {
            scopes.register(def.name.clone(), def as &mut dyn SymbolDefinition);
        }
        self.body.traverse_preorder_mut(&scopes, f)
    }
}

impl Statement for ParallelFor {}