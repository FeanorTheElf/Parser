use super::super::language::prelude::*;
use super::super::language::compiler::*;
use super::super::lexer::lexer::lex;
use super::super::parser::TopLevelParser;
use super::super::util::skip_last;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::ffi::{OsStr, OsString};
use std::io;
use std::io::BufRead;

#[derive(Debug)]
pub enum Error {
    Compiler(CompileError),
    Reader(io::Error),
    Backend(OutputError),
    External(io::Error),
    CompilerNotFound(String)
}

impl From<CompileError> for Error {
    fn from(err: CompileError) -> Self {
        Error::Compiler(err)
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Reader(err)
    }
}

impl From<OutputError> for Error {
    fn from(err: OutputError) -> Self {
        Error::Backend(err)
    }
}

pub struct FileBackend<'a> {
    compiler: Box<dyn 'a + Compiler>,
    out_extension: &'a OsStr
}

impl<'a> FileBackend<'a> {

    pub fn new(out_extension: &'a OsStr, compiler: Box<dyn 'a + Compiler>) -> Self {
        FileBackend {
            out_extension: out_extension,
            compiler: compiler
        }
    }

    fn parse_file(&mut self, path: &Path) -> Result<Program, Error> {
        let input = File::open(Path::join(path, "main.gwh"))?;
        let reader = io::BufReader::new(input);
        let mut error: Option<io::Error> = None;
        let ast_result = {
            let characters = reader.lines().map_while(|l| match l { 
                Result::Ok(v) => Some(v), 
                Result::Err(e) => { error = Some(e); None }
            }).flat_map(|l| l.chars().chain(std::iter::once('\n')).collect::<Vec<_>>().into_iter());
            let mut stream = lex(characters);
            Program::parse(&mut stream)
        };
        if let Some(e) = error {
            return Err(Error::from(e));
        }
        return Ok(ast_result?);
    }
    
    pub fn compile(&mut self, input: &Path, output: &Path) -> Result<PathBuf, Error> {
        let mut ast = self.parse_file(input)?;

        let mut out = File::create(output).map_err(OutputError::from)?;
        self.compiler.init()?;
        self.compiler.transform_program(&mut ast)?;
        let mut writer = CodeWriter::new(&mut out);
        self.compiler.generate(&ast, &mut writer)?;

        return Ok(output.to_owned());
    }
}

pub trait ExternalCompiler {
    fn run(&mut self, input: &Path, output: &Path) -> Result<(), Error>;
}

pub struct MultiStageBackend<'a> {
    backend: FileBackend<'a>,
    external_parts: Vec<Box<dyn 'a + ExternalCompiler>>
}

pub struct MultiStageBackendOptions {
    pub input: PathBuf,
    pub intermediate_dirs: Vec<PathBuf>,
    pub out_dir: PathBuf,
    pub out_name: OsString
}

impl<T> From<T> for MultiStageBackendOptions 
    where T: AsRef<Path>
{
    fn from(parent: T) -> Self {
        MultiStageBackendOptions {
            input: Path::join(parent.as_ref(), "src"),
            out_dir: Path::join(parent.as_ref(), "target"),
            out_name: OsString::from("out"),
            intermediate_dirs: vec![Path::join(parent.as_ref(), "temp"), Path::join(parent.as_ref(), "temp2")]
        }
    }
}

impl std::default::Default for MultiStageBackendOptions {
    fn default() -> Self {
        Self::from(".")
    }
}

impl<'a> MultiStageBackend<'a> {

    pub fn new(backend: FileBackend<'a>, toolchain: Vec<Box<dyn 'a + ExternalCompiler>>) -> Self {
        MultiStageBackend {
            backend: backend,
            external_parts: toolchain
        }
    }

    pub fn run(&mut self, options: MultiStageBackendOptions) -> Result<(), Error> {
        assert!(self.external_parts.len() == 0 || options.intermediate_dirs.len() > 0);
        assert!(self.external_parts.len() <= 1 || options.intermediate_dirs.len() > 1);
        if self.external_parts.len() == 0 {
            let mut out_filename = options.out_name.to_os_string();
            out_filename.push(&OsString::from(".".to_owned()));
            out_filename.push(self.backend.out_extension);
            let out_file = Path::join(&options.out_dir, out_filename);
            self.backend.compile(&options.input, &out_file)?;
        } else {
            let mut dirs = options.intermediate_dirs.iter().cycle();
            let mut out_filename = OsString::from("out.".to_owned());
            out_filename.push(self.backend.out_extension);

            let mut last_dir = dirs.next().unwrap();
            let out_file = Path::join(&last_dir, out_filename);
            self.backend.compile(&options.input, &out_file)?;
            for (dir, external_compiler) in dirs.zip(skip_last(self.external_parts.iter_mut())) {
                external_compiler.run(last_dir, dir)?;
                last_dir = dir;
            }

            self.external_parts.last_mut().unwrap().run(last_dir, &options.out_dir)?;
        }
        return Ok(());
    }
}
