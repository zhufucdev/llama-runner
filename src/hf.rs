use std::{io::IsTerminal, path::PathBuf, str::FromStr};

use hf_hub::api::tokio::ApiBuilder;

pub fn build_hf_api() -> Result<hf_hub::api::tokio::Api, hf_hub::api::tokio::ApiError> {
    let mut api = ApiBuilder::new()
        .with_progress(std::io::stdin().is_terminal())
        .with_token(std::env::var("HF_TOKEN").ok())
        .with_chunk_size(Some(2 << 28));
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        api = api.with_endpoint(endpoint);
    }
    if let Ok(cache) = std::env::var("HF_HOME") {
        api = api.with_cache_dir(
            PathBuf::from_str(&cache).expect("HF_HOME env var is not a valid path"),
        );
    }
    api.build()
}
