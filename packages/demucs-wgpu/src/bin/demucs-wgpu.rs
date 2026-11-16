use clap::Parser;

fn main() -> anyhow::Result<()> {
    let cli = demucs_wgpu::Cli::parse();
    demucs_wgpu::run(cli)
}
