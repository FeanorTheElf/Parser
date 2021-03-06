use super::super::language::prelude::*;
use super::super::cuda::*;
use super::cli_backend::*;
use super::win_cl::*;

use std::path::{Path, PathBuf};
use std::fs::File;
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::process;

macro_rules! os_string {
    ($expr:expr) => {
        singleton!({ OsString::from(($expr).to_owned()) }: OsString).as_os_str()
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NVCCOutputMode {
    EXE, DLL
}

struct NVCC {
    mode: NVCCOutputMode
}

fn quote<S>(s: S) -> OsString
    where S: AsRef<OsStr>
{
    let quotation_mark = os_string!("'");
    let mut result = quotation_mark.to_owned();
    result.push(s);
    result.push(quotation_mark);
    return result;
}

impl NVCC {
    fn run_compilation(&mut self, input_file: &Path, output_file: &Path, cl_config: &WinCLConfig) -> Result<process::Output, Error> {
        let cuda_dir_path = PathBuf::from("cuda");
        println!("Using VS Installation at {}", cl_config.installation_path().display());
        let devcmd_path = cl_config.devcmd_path();

        let cuda_path = std::env::var("CUDA_PATH")
            .map_err(|_| Error::CompilerNotFound("Could not find visual Cuda Installation (required for compiler nvcc.exe)".to_owned()))?;
        println!("Using Cuda Installation at {}", cuda_path);
        let nvcc_path = Path::join(&Path::join(&PathBuf::from(cuda_path), "bin"), "nvcc.exe");

        let mut command = process::Command::new(Path::join(&cuda_dir_path, "compile.bat"));
        return Ok(command.arg(devcmd_path).arg(nvcc_path).arg(input_file).arg(output_file)
            .stdout(std::process::Stdio::piped()).output()?);
    }
}

impl ExternalCompiler for NVCC {
    fn run(&mut self, input: &Path, output: &Path) -> Result<(), Error> {
        assert_eq!(self.mode, NVCCOutputMode::EXE);
        let cuda_dir_path = PathBuf::from("cuda");
        let input_path = Path::join(input, "main.cu");
        {
            let mut main = File::create(&input_path)?;
            write!(main, "#include \"out.cuh\"\n\nint main() {{\n    main_();\n    return 0;\n}}")?;
        }
        std::fs::copy(Path::join(&cuda_dir_path, "gwaihir.h"), Path::join(input, "gwaihir.h"))?;

        let cl_configs = WinCLConfig::query()?;
        let cl_config = WinCLConfig::preferred(&cl_configs)
            .ok_or_else(|| Error::CompilerNotFound("Could not find Visual Studio Installation (required for compiler cl.exe)".to_owned()))?;
        let output_path = Path::join(output, "out.exe");

        let compiler_output = self.run_compilation(&input_path, &output_path, cl_config)?;

        let output_log_path = Path::join(output, "log.txt");
        let mut logfile = File::create(output_log_path)?;
        write!(logfile, "{}", std::str::from_utf8(&compiler_output.stdout[..]).expect("invalid utf-8 returned by compiler"))?;
        write!(logfile, "{}", std::str::from_utf8(&compiler_output.stderr[..]).expect("invalid utf-8 returned by compiler"))?;
        return Ok(());
    }
}

fn create_cuda_file_backend() -> FileBackend<'static> {
    unimplemented!()
    //FileBackend::new(os_string!("cuh"), Box::new(CudaBackend::new()))
}

pub fn create_cuda_source_backend() -> MultiStageBackend<'static> {
    MultiStageBackend::new(create_cuda_file_backend(), vec![])
}

pub fn create_cuda_exe_backend() -> MultiStageBackend<'static> {
    MultiStageBackend::new(create_cuda_file_backend(), vec![Box::new(NVCC { mode: NVCCOutputMode::EXE })])
}