//! 严格验证: 用 gradio crate 0.4.1 的真实类型定义反序列化
//! https://voxcpm.modelbest.cn 的 /config 与 /gradio_api/info 响应,
//! 确认 `Client::new_sync` 的失败发生在哪一步、具体字段是什么。
//!
//! 类型定义从 gradio-0.4.1/src/structs.rs 原样复制 (该 crate 未导出这些类型)。

use serde::Deserialize;

#[derive(Deserialize)]
pub struct AppConfigVersionOnly {
    pub version: String,
}

#[derive(Deserialize)]
pub struct AppConfig {
    pub components: Vec<ComponentMeta>,
    pub dependencies: Vec<Dependency>,
    pub mode: String,
    pub root: String,
    pub theme: String,
    pub title: String,
    pub version: String,
    pub protocol: String,
    pub layout: serde_json::Value,
    pub auth_message: Option<String>,
    pub css: Option<String>,
    pub js: Option<String>,
    pub head: Option<String>,
    pub root_url: Option<String>,
    pub space_id: Option<String>,
    pub stylesheets: Vec<String>,
    pub path: Option<String>,
    pub theme_hash: Option<StringOrI64>,
    pub username: Option<String>,
    pub max_file_size: Option<i64>,
    pub api_prefix: Option<String>,
    #[serde(default)]
    pub auth_required: Option<bool>,
    #[serde(default)]
    pub analytics_enabled: Option<bool>,
    #[serde(default)]
    pub connect_heartbeat: Option<bool>,
    #[serde(default)]
    pub dev_mode: Option<bool>,
    #[serde(default)]
    pub enable_queue: Option<bool>,
    #[serde(default)]
    pub show_error: Option<bool>,
    #[serde(default)]
    pub is_space: Option<bool>,
    #[serde(default)]
    pub is_colab: Option<bool>,
    #[serde(default)]
    pub show_api: Option<bool>,
}

#[derive(Deserialize)]
pub struct ComponentMeta {
    pub r#type: String,
    pub id: StringOrI64,
    pub props: serde_json::Value,
    #[serde(default)]
    pub component_class_id: String,
    pub component: Option<serde_json::Value>,
    pub value: Option<serde_json::Value>,
    pub key: Option<String>,
}

#[derive(Deserialize)]
pub struct Dependency {
    pub api_name: String,
    #[serde(default = "default_id")]
    pub id: i64,
    pub queue: Option<bool>,
}

fn default_id() -> i64 {
    -1
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum StringOrI64 {
    String(String),
    I64(i64),
}

#[derive(Deserialize)]
pub struct ApiInfo {
    pub named_endpoints: std::collections::HashMap<String, EndpointInfo>,
}

#[derive(Deserialize)]
pub struct EndpointInfo {
    pub parameters: Vec<ApiData>,
    pub returns: Vec<ApiData>,
    #[serde(default)]
    pub show_api: Option<bool>,
}

#[derive(Deserialize)]
pub struct ApiData {
    pub label: Option<String>,
    pub parameter_name: Option<String>,
    pub parameter_default: Option<serde_json::Value>,
    pub parameter_has_default: Option<bool>,
    pub component: String,
    pub example_input: Option<serde_json::Value>,
    pub r#type: ApiDataType,
    pub python_type: ApiDataPythonType,
}

#[derive(Deserialize)]
pub struct ApiDataType {
    pub r#type: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Deserialize)]
pub struct ApiDataPythonType {
    pub r#type: String,
    pub description: String,
}

fn main() -> anyhow::Result<()> {
    let http = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // ---- /config: 版本检查 + AppConfig 全量反序列化 ----
    let config_raw = http
        .get("https://voxcpm.modelbest.cn/config")
        .send()?
        .text()?;
    let v: AppConfigVersionOnly = serde_json::from_str(&config_raw)?;
    println!("[config] version = {}", v.version);
    match serde_json::from_str::<AppConfig>(&config_raw) {
        Ok(_) => println!("[config] AppConfig 反序列化 OK ({} bytes)", config_raw.len()),
        Err(e) => println!("[config] AppConfig 失败: {e}"),
    }

    // ---- /gradio_api/info: ApiInfo 反序列化 ----
    let info_raw = http
        .get("https://voxcpm.modelbest.cn/gradio_api/info")
        .send()?
        .text()?;
    println!(
        "[info] 原始响应 col 60-80: {:?}",
        &info_raw.chars().skip(60).take(20).collect::<String>()
    );
    match serde_json::from_str::<ApiInfo>(&info_raw) {
        Ok(info) => println!(
            "[info] ApiInfo 反序列化 OK ({} named endpoints)",
            info.named_endpoints.len()
        ),
        Err(e) => println!("[info] ApiInfo 失败: {e}"),
    }

    Ok(())
}
