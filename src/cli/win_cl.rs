
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::str;

use super::cli_backend::Error;

#[allow(non_snake_case)]
#[derive(Debug, Deserialize)]
pub struct WinCLConfig {
    installationPath: String,
    installationVersion: String
}

impl WinCLConfig {

    pub fn major(&self) -> u32 {
        let index = self.installationVersion.find('.').unwrap();
        self.installationVersion[0..index].parse::<u32>().unwrap()
    }

    pub fn minor(&self) -> u32 {
        let start_index = self.installationVersion.find('.').unwrap() + 1;
        let end_index = self.installationVersion[start_index..].find('.').unwrap();
        self.installationVersion[start_index..end_index].parse::<u32>().unwrap()
    }

    pub fn installation_path(&self) -> PathBuf {
        PathBuf::from(&self.installationPath)
    }

    pub fn devcmd_path(&self) -> PathBuf {
        Path::join(&self.installation_path(), "Common7\\Tools\\VsDevCmd.bat")
    }

    pub fn cl_path(&self) -> PathBuf {
        unimplemented!()
    }

    fn vswhere_at<P>(path: P) -> Result<Vec<WinCLConfig>, Error> 
        where P: AsRef<Path>
    {
        let mut cmd = process::Command::new(path.as_ref());
        let output = cmd.args(&["-format", "json", "-utf8"]).output()?;
        assert!(output.status.success());
        let json = str::from_utf8(&output.stdout).expect("not utf-8 from vswhere");
        Ok(serde_json::from_str(json).expect("invalid json from vswhere"))
    }

    pub fn query() -> Result<Vec<WinCLConfig>, Error> {
        let program_files_x86 = std::env::var("ProgramFiles(x86)").map(PathBuf::from)
            .map_err(|_| Error::CompilerNotFound("Environment variable 'ProgramFiles(x86)' is not defined".to_owned()));
        let vswhere_result = if let Ok(data) = Self::vswhere_at("vswhere.exe") { 
            data 
        } else {
            if let Ok(data) = Self::vswhere_at("vswhere.exe") { 
                data 
            } else if let Ok(programs_x86) = program_files_x86 {
                if let Ok(data) = Self::vswhere_at(Path::join(&programs_x86, "Microsoft Visual Studio\\Installer\\vswhere.exe")) {
                    data
                } else {
                    Err(Error::CompilerNotFound("Could not find vswhere.exe to choose visual studio installation".to_owned()))?
                }
            } else {
                Err(Error::CompilerNotFound("Could not find vswhere.exe to choose visual studio installation".to_owned()))?
            }
        };
        return Ok(vswhere_result);
    }

    pub fn preferred(options: &Vec<WinCLConfig>) -> Option<&WinCLConfig> {
        options.iter().fold(None, |current: Option<&WinCLConfig>, next: &WinCLConfig| {
            if let Some(current) = current {
                if (current.major() > next.major()) || (current.minor() > next.minor()) {
                    Some(current)
                } else {
                    Some(next)
                }
            } else {
                Some(next)
            }
        })
    }
}