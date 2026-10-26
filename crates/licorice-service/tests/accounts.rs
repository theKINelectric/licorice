//! Storage/authorization fixtures seed sessions directly; real WebAuthn is tested in Chrome separately.
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use licorice::rusqlite::{Connection, params};
use licorice_service::{Config, Service};
use serde_json::{Value, json};
use tower::ServiceExt;
const ORIGIN: &str = "http://localhost:8080";
fn config(p: &std::path::Path) -> Config {
    let mut c = Config::new(p, ORIGIN);
    c.secure_cookie = false;
    c.accounts_enabled = true;
    c
}
fn db(p: &std::path::Path) -> Connection {
    Connection::open(p.join("durable/accounts.sqlite")).unwrap()
}
fn hash(v: &str) -> String {
    blake3::hash(v.as_bytes()).to_hex().to_string()
}
fn seed(p: &std::path::Path, n: char) -> (String, String, String) {
    let who = n.to_string().repeat(64);
    let token = (if n == 'a' { "c" } else { "d" }).repeat(64);
    let recovery = if n == 'a' { "e" } else { "f" }.repeat(64);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
    let d = db(p);
    d.execute(
        "INSERT INTO accounts VALUES(?1,?2,?3,?4,'active',?5)",
        params![who, format!("account-{n}"), "00000000-0000-4000-8000-000000000000", hash(&recovery), now],
    )
    .unwrap();
    d.execute("INSERT INTO sessions VALUES(?1,?2,?3,?4)", params![hash(&token), who, now, now + 604800000]).unwrap();
    (who, format!("licorice_account={token}"), recovery)
}
async fn call(
    app: &Router,
    method: &str,
    path: &str,
    credential: Option<&str>,
    body: Option<Value>,
) -> (StatusCode, Value, Option<String>) {
    let mut r = Request::builder().method(method).uri(path);
    if let Some(c) = credential {
        if c.contains('=') {
            r = r.header("cookie", c).header("origin", ORIGIN)
        } else {
            r = r.header("authorization", format!("Bearer {c}"))
        }
    } else {
        r = r.header("origin", ORIGIN)
    };
    if body.is_some() {
        r = r.header("content-type", "application/json")
    };
    let r = app
        .clone()
        .oneshot(r.body(Body::from(body.map(|v| v.to_string()).unwrap_or_default())).unwrap())
        .await
        .unwrap();
    let status = r.status();
    let cookie = r.headers().get_all("set-cookie").iter().find_map(|v| {
        let v = v.to_str().unwrap();
        v.starts_with("licorice_account=").then(|| v.split(';').next().unwrap().to_owned())
    });
    let bytes = to_bytes(r.into_body(), 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null), cookie)
}
async fn create(app: &Router, c: &str, id: &str) -> String {
    let (s, v, _) = call(app, "POST", "/api/v2/projects", Some(c), Some(json!({"request_id":id,"name":id}))).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v["project_id"].as_str().unwrap().into()
}
async fn key(app: &Router, c: &str, p: &str, perm: &str) -> Value {
    let (s, v, _) = call(
        app,
        "POST",
        &format!("/api/v2/projects/{p}/keys"),
        Some(c),
        Some(json!({"name":"test","permission":perm})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v
}
#[tokio::test]
async fn namespace_scopes_attribution_and_replay() {
    let tmp = tempfile::tempdir().unwrap();
    let s = Service::open(config(tmp.path())).unwrap();
    let (_, a, _) = seed(tmp.path(), 'a');
    let (_, b, _) = seed(tmp.path(), 'b');
    let app = s.router();
    let p = create(&app, &a, "first").await;
    assert_eq!(create(&app, &a, "first").await, p);
    assert_eq!(
        call(&app, "POST", "/api/v2/projects", Some(&a), Some(json!({"request_id":"first","name":"different"})))
            .await
            .0,
        StatusCode::CONFLICT
    );
    let path = format!("/api/v2/projects/{p}");
    assert_eq!(call(&app, "GET", &path, Some(&b), None).await.0, StatusCode::NOT_FOUND);
    let p2 = create(&app, &b, "second").await;
    let r = key(&app, &a, &p, "read").await;
    let w = key(&app, &a, &p, "write").await;
    let read = r["api_key"].as_str().unwrap();
    let write = w["api_key"].as_str().unwrap();
    assert_eq!(call(&app, "GET", &path, Some(read), None).await.0, StatusCode::OK);
    assert_eq!(
        call(
            &app,
            "POST",
            &format!("{path}/observations"),
            Some(read),
            Some(json!({"request_id":"one","text":"blue plum"}))
        )
        .await
        .0,
        StatusCode::FORBIDDEN
    );
    for endpoint in ["/api/v2/account".to_owned(), format!("{path}/keys"), format!("/api/v2/projects/{p2}")] {
        assert_ne!(call(&app, "GET", &endpoint, Some(write), None).await.0, StatusCode::OK)
    }
    let input = json!({"request_id":"one","text":"The launch snack is blue plum."});
    let (status, first, _) =
        call(&app, "POST", &format!("{path}/observations"), Some(write), Some(input.clone())).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(call(&app, "POST", &format!("{path}/observations"), Some(write), Some(input)).await.1, first);
    assert_eq!(
        call(
            &app,
            "POST",
            &format!("{path}/observations"),
            Some(write),
            Some(json!({"request_id":"one","text":"different"}))
        )
        .await
        .0,
        StatusCode::CONFLICT
    );
    let (_, state, _) = call(&app, "GET", &path, Some(&a), None).await;
    assert_eq!(state["memory"]["observations"][0]["observation"]["origin"], "imported_document");
    assert!(state["memory"]["claims"].as_array().unwrap().is_empty());
    let (_, out, _) =
        call(&app, "POST", &format!("{path}/recall"), Some(read), Some(json!({"query":"launch snack"}))).await;
    assert!(out["text"].as_str().unwrap().contains("blue plum"));
    assert!(out["usage"]["tokens"].as_u64().unwrap() <= 2000);
    let (_, export, _) = call(&app, "GET", "/api/v2/account/export", Some(&a), None).await;
    assert!(!export.to_string().contains(write));
    assert!(!export.to_string().contains(read));
    let dump = db(tmp.path())
        .query_row("SELECT key_hash FROM project_keys WHERE id=?1", [w["key_id"].as_str().unwrap()], |r| {
            r.get::<_, String>(0)
        })
        .unwrap();
    assert_eq!(dump, hash(write));
}
#[tokio::test]
async fn restart_retention_revocation_and_tombstones() {
    let tmp = tempfile::tempdir().unwrap();
    let s = Service::open(config(tmp.path())).unwrap();
    let (_, a, _) = seed(tmp.path(), 'a');
    let app = s.router();
    let p = create(&app, &a, "durable").await;
    let path = format!("/api/v2/projects/{p}");
    let w = key(&app, &a, &p, "write").await;
    let token = w["api_key"].as_str().unwrap();
    assert_eq!(
        call(
            &app,
            "POST",
            &format!("{path}/observations"),
            Some(token),
            Some(json!({"request_id":"one","text":"Remember the violet launch."}))
        )
        .await
        .0,
        StatusCode::OK
    );
    call(&app, "POST", &format!("{path}/sessions"), Some(token), Some(json!({"request_id":"session"}))).await;
    drop(app);
    drop(s);
    let s = Service::open(config(tmp.path())).unwrap();
    let app = s.router();
    assert_eq!(
        call(&app, "GET", &path, Some(token), None).await.1["memory"]["observations"].as_array().unwrap().len(),
        1
    );
    let (_, v, _) = call(&app, "POST", &format!("{path}/retention"), Some(&a), Some(json!({"retention_days":1}))).await;
    assert!(v["delete_at"].as_i64().is_some());
    call(&app, "POST", &format!("{path}/retention"), Some(&a), Some(json!({"retention_days":null}))).await;
    assert!(call(&app, "GET", &path, Some(&a), None).await.1["project"]["delete_at"].is_null());
    db(tmp.path()).execute("UPDATE project_keys SET expires_at=0 WHERE project_id=?1", [&p]).unwrap();
    assert_eq!(call(&app, "GET", &path, Some(token), None).await.0, StatusCode::UNAUTHORIZED);
    db(tmp.path()).execute("UPDATE projects SET delete_at=0 WHERE id=?1", [&p]).unwrap();
    assert_eq!(call(&app, "GET", &path, Some(&a), None).await.0, StatusCode::NOT_FOUND);
    s.cleanup_durable().await.unwrap();
    assert!(!tmp.path().join("durable/projects").join(&p).exists());
    drop(app);
    drop(s);
    let s = Service::open(config(tmp.path())).unwrap();
    assert_eq!(call(&s.router(), "GET", &path, Some(&a), None).await.0, StatusCode::NOT_FOUND);
    assert_eq!(
        db(tmp.path()).query_row("SELECT state FROM projects WHERE id=?1", [&p], |r| r.get::<_, String>(0)).unwrap(),
        "deleted"
    );
}
#[tokio::test]
async fn recovery_revokes_credentials_and_account_deletion_survives_restart() {
    let tmp = tempfile::tempdir().unwrap();
    let s = Service::open(config(tmp.path())).unwrap();
    let (_, a, recovery) = seed(tmp.path(), 'a');
    let app = s.router();
    let p = create(&app, &a, "project").await;
    let w = key(&app, &a, &p, "write").await;
    let token = w["api_key"].as_str().unwrap();
    let (status, v, cookie) =
        call(&app, "POST", "/api/v2/auth/recover", None, Some(json!({"username":"account-a","recovery_key":recovery})))
            .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let new = cookie.unwrap();
    assert_eq!(call(&app, "GET", "/api/v2/account", Some(&a), None).await.0, StatusCode::UNAUTHORIZED);
    assert_eq!(
        call(&app, "GET", &format!("/api/v2/projects/{p}"), Some(token), None).await.0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        call(&app, "POST", "/api/v2/auth/recover", None, Some(json!({"username":"account-a","recovery_key":recovery})))
            .await
            .0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(call(&app, "DELETE", "/api/v2/account", Some(&new), None).await.0, StatusCode::OK);
    assert!(!tmp.path().join("durable/projects").join(&p).exists());
    drop(app);
    drop(s);
    let app = Service::open(config(tmp.path())).unwrap().router();
    assert_eq!(call(&app, "GET", "/api/v2/account", Some(&new), None).await.0, StatusCode::UNAUTHORIZED);
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v2/auth/recover",
            None,
            Some(json!({"username":"account-a","recovery_key":v["recovery_key"]}))
        )
        .await
        .0,
        StatusCode::UNAUTHORIZED
    );
}
#[tokio::test]
async fn challenge_csrf_and_disabled_boundaries() {
    let tmp = tempfile::tempdir().unwrap();
    let app = Service::open(config(tmp.path())).unwrap().router();
    let (s, v, _) = call(&app, "POST", "/api/v2/auth/register/start", None, Some(json!({"username":"new-user"}))).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["publicKey"]["authenticatorSelection"]["userVerification"], "required");
    assert_eq!(db(tmp.path()).query_row("SELECT count(*) FROM accounts", [], |r| r.get::<_, i64>(0)).unwrap(), 0);
    let req = Request::builder()
        .method("POST")
        .uri("/api/v2/auth/register/start")
        .header("origin", "https://evil.invalid")
        .header("content-type", "application/json")
        .body(Body::from("{\"username\":\"intruder\"}"))
        .unwrap();
    assert_eq!(app.clone().oneshot(req).await.unwrap().status(), StatusCode::FORBIDDEN);
    assert_ne!(call(&app, "POST", "/api/v2/auth/register/finish", None, Some(json!({}))).await.0, StatusCode::OK);
    let tmp2 = tempfile::tempdir().unwrap();
    let mut c = config(tmp2.path());
    c.accounts_enabled = false;
    assert_eq!(
        call(&Service::open(c).unwrap().router(), "GET", "/api/v2/account", None, None).await.0,
        StatusCode::NOT_FOUND
    );
}
