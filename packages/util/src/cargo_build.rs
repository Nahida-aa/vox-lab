use std::path::{Path, PathBuf};

/// workspace 根目录: 上溯两层 (packages/util -> vox-lab)。
fn workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

/// 构造 `cargo build -p <package> --bin <bin>` 命令。
///
/// `features` 非空时追加 `--no-default-features --features <a>,<b>`, 确保只启用目标后端
/// (各 burn 包 `default = ["wgpu"]`, 不关 default 会让 wgpu 与指定后端同时编译,
/// 触发 `main.rs` 中 `type B` 重复定义冲突)。`release` 为真时追加 `--release`。
///
/// 注: 不在此处 `.output()`, 交由调用方决定同步/流式, 故只返回构造好的 [`std::process::Command`]。
pub fn cargo_build_cmd(
    package: &str,
    bin: &str,
    features: &[&str],
    release: bool,
) -> std::process::Command {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut cmd = std::process::Command::new(cargo);
    cmd.arg("build")
        .arg("-p")
        .arg(package)
        .arg("--bin")
        .arg(bin);
    if release {
        cmd.arg("--release");
    }
    if !features.is_empty() {
        // 关掉 default features, 只启用目标后端, 避免多后端 type B 冲突
        cmd.arg("--no-default-features");
        let joined = features.join(",");
        cmd.arg("--features").arg(joined);
    }
    cmd
}

/// 执行 `cargo build [-p <package>] --bin <bin> [--release] [--no-default-features --features ...]`,
/// 成功后在 `target/release` / `target/debug` 中定位产物 (优先 release)。失败 (编译错误 /
/// 产物缺失) 时返回错误, 由调用方决定如何上报。
///
/// `features` 用于指定后端 feature (如 demucs-burn 的 `tch`/`wgpu`/`cuda`...), 因为各 burn 包
/// 用 `required-features` 把二进制绑定到对应 feature, 不显式 `--features` 会编译失败; 同时会
/// 关掉 default features (默认 wgpu), 避免多后端 `type B` 重复定义冲突。`release` 控制是否
/// 编 release 产物 (与 `find_release_bin` 的 release 优先、及手动提示一致)。
///
/// 编译输出 (stdout/stderr) 直接 inherit 到当前进程, 便于实时看到 `Compiling...` /
/// 进度, 避免长时间静默让人误以为卡住。失败时仍由 exit status 判断并给出手动命令提示。
pub fn cargo_build_bin(
    package: &str,
    bin: &str,
    features: &[&str],
    release: bool,
) -> anyhow::Result<PathBuf> {
    let feat_str = if features.is_empty() {
        String::new()
    } else {
        format!(" --no-default-features --features {}", features.join(","))
    };
    let rel_str = if release { " --release" } else { "" };
    let cmdline = format!("cargo build -p {package} --bin {bin}{rel_str}{feat_str}");
    tracing::info!("[auto-build] 未找到 {bin}, 执行: {cmdline}");
    let mut cmd = cargo_build_cmd(package, bin, features, release);
    // 透传编译输出, 实时可见
    cmd.stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("无法执行 `{cmdline}` (cargo 未安装/未找到?): {e}"))?;
    if !status.success() {
        return Err(anyhow::anyhow!(
            "编译 {bin} 失败 (`{cmdline}`), 详见上方 cargo 输出"
        ));
    }
    // 优先 release, 回退 debug (dev 构建)
    let repo = workspace_root();
    for profile in ["release", "debug"] {
        let p = repo.join("target").join(profile).join(bin);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(anyhow::anyhow!(
        "编译 {bin} 成功但未找到产物 (target/release 或 target/debug 下均无 {bin})"
    ))
}