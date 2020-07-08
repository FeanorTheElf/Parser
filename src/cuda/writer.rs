pub use std::io::Write;
use std::io::Result;
use std::str;

pub struct StringWriter<'a> {
    out: &'a mut String
}

impl<'a> Write for StringWriter<'a>
{
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        let result = <&'a mut String as std::fmt::Write>::write_str(&mut self.out, str::from_utf8(buf).unwrap());
        if result.is_err() {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, result.unwrap_err()));
        } else {
            return Ok(buf.len());
        }
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

impl<'a> StringWriter<'a> {
    pub fn new(target: &'a mut String) -> StringWriter<'a> {
        StringWriter {
            out: target
        }
    }
}

pub struct CodeWriter<'a> {
    out: &'a mut (dyn Write + 'a),
    indent: String
}

impl<'a> CodeWriter<'a> {
    pub fn new<T: Write + 'a>(out: &'a mut T) -> CodeWriter<'a> {
        CodeWriter {
            out: out,
            indent: "".to_owned()
        }
    }

    pub fn enter_indented_level(&mut self) -> Result<()> {
        self.indent.push(' ');
        self.indent.push(' ');
        self.indent.push(' ');
        self.indent.push(' ');
        self.newline()
    }

    pub fn exit_indented_level(&mut self) -> Result<()> {
        self.indent.pop();
        self.indent.pop();
        self.indent.pop();
        self.indent.pop();
        self.newline()
    }

    pub fn newline(&mut self) -> Result<()> {
        write!(self.out, "\n{}", self.indent)
    }

    pub fn enter_block(&mut self) -> Result<()> {
        write!(self.out, "{{")?;
        self.enter_indented_level()
    }

    pub fn exit_block(&mut self) -> Result<()> {
        write!(self.out, "\n}}")?;
        self.exit_indented_level()
    }

    pub fn write_separated<I, G, E>(&mut self, mut it: I, mut separator: G) -> std::prelude::v1::Result<(), E> 
        where I: Iterator, 
            I::Item: FnOnce(&mut CodeWriter<'a>) -> std::prelude::v1::Result<(), E>,
            G: FnMut(&mut CodeWriter<'a>) -> std::prelude::v1::Result<(), E>,
            E: From<std::io::Error>
    {
        if let Some(value) = it.next() {
            value(self)?;
        }
        for value in it {
            separator(self)?;
            value(self)?;
        }
        Ok(())
    }

    pub fn write_comma_separated<I, E>(&mut self, it: I) -> std::prelude::v1::Result<(), E> 
        where I: Iterator, 
            I::Item: FnOnce(&mut CodeWriter<'a>) -> std::prelude::v1::Result<(), E>,
            E: From<std::io::Error>
    {
        self.write_separated(it, |out| write!(out, ", ").map_err(E::from))
    }
}

impl<'a> Write for CodeWriter<'a> {
    
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        self.out.write(buf)
    }

    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> Result<usize> {
        self.out.write_vectored(bufs)
    }

    fn flush(&mut self) -> Result<()> {
        self.out.flush()
    }

    fn write_all(&mut self, mut buf: &[u8]) -> Result<()> {
        self.out.write_all(buf)
    }

    fn write_fmt(&mut self, fmt: std::fmt::Arguments<'_>) -> Result<()> {
        self.out.write_fmt(fmt)
    }
}