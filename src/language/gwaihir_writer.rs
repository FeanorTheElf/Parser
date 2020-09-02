use super::prelude::*;
use super::backend::*;

pub trait AstWriter {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError>;
}

fn get_priority(expr: &Expression) -> i32 {
    match expr {
        Expression::Call(call) => match &call.function {
            Expression::Call(subcall) => get_priority(&call.function),
            Expression::Literal(_) => panic!("Literal not callable"),
            Expression::Variable(var) => match &var.identifier {
                Identifier::Name(name) => i32::MAX,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionIndex) => i32::MAX,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionMul) | 
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionUnaryDiv) => 1,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionAdd) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionUnaryNeg) => 0,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionEq) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionNeq) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionLs) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionGt) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionLeq) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionGeq) => -1,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionAnd) => -2,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionOr) => -3,
            }
        },
        Expression::Literal(_) => i32::MAX,
        Expression::Variable(_) => i32::MAX
    }
}

fn write_non_builtin_call(call: &FunctionCall, function: &Expression, out: &mut CodeWriter) -> Result<(), OutputError> {
    write_expression(&call.function, i32::MAX - 1, out)?;
    write!(out, "(")?;
    for param in &call.parameters {
        write_expression(param, i32::MIN, out)?;
        write!(out, ", ")?;
    }
    write!(out, ")")?;
    return Ok(());
}

fn write_builtin_call(call: &FunctionCall, priority: i32, op: BuiltInIdentifier, out: &mut CodeWriter) -> Result<(), OutputError> {
    match op {
        BuiltInIdentifier::FunctionIndex => {
            write_expression(&call.parameters[0], i32::MAX - 1, out)?;
            write!(out, "[")?;
            for index in call.parameters.iter().skip(1) {
                write_expression(index, i32::MIN, out)?;
                write!(out, ", ")?;
            }
            write!(out, "]")?;
        },
        BuiltInIdentifier::FunctionUnaryDiv | BuiltInIdentifier::FunctionUnaryNeg => {
            let symbol = op.get_symbol();
            write!(out, "{}", symbol)?;
            debug_assert_eq!(call.parameters.len(), 1);
            write_expression(&call.parameters[0], priority, out)?;
        },
        op => {
            let symbol = op.get_symbol();
            out.write_separated(call.parameters.iter().map(|p| move |out: &mut CodeWriter| write_expression(p, priority, out)), |out| write!(out, "{}", symbol).map_err(OutputError::from))?;
        }
    };
    return Ok(());
}

fn write_expression(expr: &Expression, parent_priority: i32, out: &mut CodeWriter) -> Result<(), OutputError> {
    if get_priority(expr) <= parent_priority {
        write!(out, "(")?;
    }
    match expr {
        Expression::Call(call) => match &call.function {
            Expression::Variable(var) => match &var.identifier {
                Identifier::Name(_) => write_non_builtin_call(&**call, &call.function, out)?,
                Identifier::BuiltIn(op) => write_builtin_call(call, get_priority(expr), *op, out)?
            },
            func => write_non_builtin_call(&**call, func, out)?
        },
        Expression::Literal(lit) => write!(out, "{}", lit.value)?,
        Expression::Variable(var) => match &var.identifier {
            Identifier::Name(name) => write!(out, "{}", name)?,
            Identifier::BuiltIn(op) => write!(out, "{}", op.get_symbol())?
        }
    };
    if get_priority(expr) <= parent_priority {
        write!(out, ")")?;
    }
    return Ok(());
}

impl AstWriter for Expression {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write_expression(self, i32::MIN, out)
    }
}

impl AstWriter for If {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "if (")?;
        write_expression(&self.condition, i32::MIN, out)?;
        write!(out, ")")?;
        self.body.write(out)?;
        return Ok(());
    }
}

impl AstWriter for While {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "while (")?;
        write_expression(&self.condition, i32::MIN, out)?;
        write!(out, ")")?;
        self.body.write(out)?;
        return Ok(());
    }
}

impl AstWriter for Return {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "return")?;
        if let Some(val) = &self.value {
            write!(out, " ")?;
            write_expression(val, i32::MIN, out)?;
        }
        write!(out, ";")?;
        return Ok(());
    }
}

impl AstWriter for dyn Statement {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
    if let Some(statement) = self.dynamic().downcast_ref::<If>() {
        statement.write(out)
    } else if let Some(statement) = self.dynamic().downcast_ref::<While>() {
        statement.write(out)
    } else if let Some(statement) = self.dynamic().downcast_ref::<Block>() {
        statement.write(out)
    } else if let Some(statement) = self.dynamic().downcast_ref::<Return>() {
        statement.write(out)
    } else if let Some(statement) = self.dynamic().downcast_ref::<LocalVariableDeclaration>() {
        statement.write(out)
    } else if let Some(statement) = self.dynamic().downcast_ref::<Assignment>() {
        statement.write(out)
    } else if let Some(statement) = self.dynamic().downcast_ref::<Goto>() {
        statement.write(out)
    } else if let Some(statement) = self.dynamic().downcast_ref::<Label>() {
        statement.write(out)
    } else {
        panic!("Unknown statement type: {:?}", self)
    }
}
}

impl AstWriter for Block {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
    out.enter_block()?;
    out.write_separated(self.statements.iter().map(|s| move |out: &mut CodeWriter| s.write(out)), |out| out.newline().map_err(OutputError::from))?;
    out.exit_block()?;
    return Ok(());
}
}

impl AstWriter for LocalVariableDeclaration {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "let {}: {}", self.declaration.variable, self.declaration.variable_type)?;
        if let Some(val) = &self.value {
            write!(out, " = ")?;
            write_expression(val, i32::MIN, out)?;
        }
        write!(out, ";")?;
        return Ok(());
    }
}

impl AstWriter for Assignment {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write_expression(&self.assignee, i32::MIN, out)?;
        write!(out, " = ")?;
        write_expression(&self.value, i32::MIN, out)?;
        return Ok(());
    }
}

impl AstWriter for Label {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "@{}:", self.label)?;
        return Ok(());
    }
}

impl AstWriter for Goto {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "goto {};", self.target)?;
        return Ok(());
    }
}

impl AstWriter for Function {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "fn {}(", self.identifier)?;
        for param in &self.params {
            write!(out, "{}: {}, ", param.variable, param.variable_type)?;
        }
        write!(out, ")")?;
        if let Some(return_type) = &self.return_type {
            write!(out, ": {} ", return_type)?;
        } else {
            write!(out, " ")?;
        }
        if let Some(body) = &self.body {
            body.write(out)?;
        } else {
            write!(out, "native;")?;
        }
        return Ok(());
    }
}

impl AstWriter for Program {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        for item in &self.items {
            item.write(out)?;
            out.newline()?;
        }
        return Ok(());
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DisplayWrapper<'a, T: AstWriter + ?Sized> {
    content: &'a T
}

impl<'a, T: AstWriter + ?Sized> std::fmt::Display for DisplayWrapper<'a, T> {
    fn fmt<'b, 'c>(&self, fmt: &'b mut std::fmt::Formatter<'c>) -> std::fmt::Result {
        let mut writer = FormatterWriter::new(fmt);
        let mut out: CodeWriter = CodeWriter::new(&mut writer);
        self.content.write(&mut out).map_err(|_| std::fmt::Error);
        return Ok(());
    }
}

impl<'a, T: AstWriter + ?Sized> From<&'a T> for DisplayWrapper<'a, T> {
    fn from(writable: &'a T) -> DisplayWrapper<'a, T> {
        DisplayWrapper {
            content: writable
        }
    }
}