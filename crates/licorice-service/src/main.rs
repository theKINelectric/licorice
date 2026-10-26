use licorice_providers::embed::{LocalOnnx, WordLlama};
use licorice_service::{Config, Service};
use std::{path::PathBuf, sync::Arc, time::Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|a| a == "--healthcheck") {
        use std::io::{Read, Write};
        let address = std::env::var("LICORICE_LISTEN")
            .unwrap_or_else(|_| "127.0.0.1:8080".into())
            .replace("0.0.0.0:", "127.0.0.1:");
        let mut socket = std::net::TcpStream::connect_timeout(&address.parse()?, Duration::from_secs(3))?;
        socket.set_read_timeout(Some(Duration::from_secs(3)))?;
        socket.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
        let mut response = String::new();
        socket.read_to_string(&mut response)?;
        if !response.starts_with("HTTP/1.1 200 ") {
            return Err("health endpoint did not return 200".into());
        }
        return Ok(());
    }
    let origin =
        std::env::var("LICORICE_PUBLIC_ORIGIN").map_err(|_| "LICORICE_PUBLIC_ORIGIN must be set explicitly")?;
    let root = std::env::var_os("LICORICE_DATA_DIR").map(PathBuf::from).unwrap_or_else(|| "data/hosted-trial".into());
    let mut config = Config::new(root, origin);
    if let Ok(value) = std::env::var("LICORICE_SECURE_COOKIE") {
        config.secure_cookie = match value.as_str() {
            "true" => true,
            "false" => false,
            _ => return Err("LICORICE_SECURE_COOKIE must be true or false".into()),
        };
    }
    if let Some(path) = std::env::var_os("LICORICE_WORDLLAMA_DIR") {
        if std::env::var_os("LICORICE_EMBEDDING_ORIGIN").is_some() {
            return Err("select one local embedding backend".into());
        }
        config.embedder = Some(Arc::new(WordLlama::load(&PathBuf::from(path))?));
    }
    if let Ok(origin) = std::env::var("LICORICE_EMBEDDING_ORIGIN") {
        let model = std::env::var("LICORICE_EMBEDDING_MODEL")?;
        let dimensions = std::env::var("LICORICE_EMBEDDING_DIMENSIONS")?.parse()?;
        if model != "sentence-transformers/all-MiniLM-L6-v2" || dimensions != 384 {
            return Err("only the qualified 384-dimensional MiniLM profile is supported by this launcher".into());
        }
        config.representation_token_limit = Some(256);
        config.embedder = Some(Arc::new(LocalOnnx::new(&origin, &model, dimensions, "", "")?));
    }
    if let Ok(value) = std::env::var("LICORICE_MINIMUM_COSINE") {
        config.semantic.minimum_cosine = value.parse()?;
    }
    if let Ok(value) = std::env::var("LICORICE_ACCOUNTS_ENABLED") {
        config.accounts_enabled = match value.as_str() {
            "true" => true,
            "false" => false,
            _ => return Err("LICORICE_ACCOUNTS_ENABLED must be true or false".into()),
        };
    }
    let service = Service::open(config)?;
    let cleanup = service.clone();
    let maintenance = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(60));
        loop {
            tick.tick().await;
            let _ = cleanup.cleanup_expired().await;
            let _ = cleanup.cleanup_durable().await;
        }
    });
    let address = std::env::var("LICORICE_LISTEN").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener = tokio::net::TcpListener::bind(&address).await?;
    eprintln!("Licorice trial listening; native memory; no request contents are logged.");
    axum::serve(listener, service.router()).with_graceful_shutdown(shutdown()).await?;
    maintenance.abort();
    Ok(())
}

async fn shutdown() {
    #[cfg(unix)]
    {
        let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler must install");
        tokio::select! { _=tokio::signal::ctrl_c()=>{}, _=terminate.recv()=>{} }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
