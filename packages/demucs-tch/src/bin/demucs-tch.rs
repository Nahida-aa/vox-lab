use clap::Parser;

fn main() -> anyhow::Result<()> {
    let cli = demucs_tch::Cli::parse();
    demucs_tch::run(cli)
}
