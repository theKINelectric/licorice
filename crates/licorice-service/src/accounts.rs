//! Durable, passkey-authenticated account/project namespace. Preview stores are separate.
use super::*;
use axum::extract::Path as RoutePath;
use std::collections::HashMap;
use webauthn_rs::prelude::*;

const ACCOUNT_COOKIE: &str = "licorice_account";
const FLOW_COOKIE: &str = "licorice_flow";
const SESSION_MS: i64 = 7 * 86_400_000;
const FLOW_MS: i64 = 120_000;
const MAX_ACCOUNTS: i64 = 100;
const MAX_PROJECTS: i64 = 5;
const MAX_PROJECT_CREATIONS: i64 = 1000;
pub(super) struct Accounts {
    db: StdMutex<Connection>,
    webauthn: Webauthn,
    flows: StdMutex<HashMap<String, (i64, Flow)>>,
    pub(super) gate: Mutex<()>,
}
enum Flow {
    Register { name: String, account: String, uuid: Uuid, state: PasskeyRegistration, existing: bool },
    Login { account: String, state: PasskeyAuthentication },
}
fn disabled() -> ApiError {
    ApiError(StatusCode::NOT_FOUND, "accounts_disabled", "Durable accounts are not enabled.")
}
fn denied() -> ApiError {
    ApiError(StatusCode::UNAUTHORIZED, "sign_in_required", "Sign in to your account.")
}
fn forbidden() -> ApiError {
    ApiError(StatusCode::FORBIDDEN, "permission_denied", "This credential cannot perform that operation.")
}
fn missing() -> ApiError {
    ApiError(StatusCode::NOT_FOUND, "project_not_found", "Project not found or unavailable.")
}
fn auth_error() -> ApiError {
    ApiError(StatusCode::UNAUTHORIZED, "authentication_failed", "Authentication failed or expired. Start again.")
}
fn accounts(s: &Service) -> Result<&Accounts> {
    s.0.accounts.as_ref().ok_or_else(disabled)
}
fn hash(v: &str) -> String {
    blake3::hash(v.as_bytes()).to_hex().to_string()
}
fn raw_cookie(h: &HeaderMap, name: &str) -> Option<String> {
    h.get(header::COOKIE)?
        .to_str()
        .ok()?
        .split(';')
        .map(str::trim)
        .find_map(|x| x.strip_prefix(&format!("{name}=")).map(str::to_owned))
}
fn cookie_name(s: &Service, name: &str) -> String {
    if s.0.config.secure_cookie { format!("__Host-{name}") } else { name.into() }
}
fn set_cookie(s: &Service, response: &mut Response, name: &str, value: &str, seconds: i64) -> Result<()> {
    let value = format!(
        "{}={value}; Path=/; HttpOnly; SameSite=Strict; Max-Age={seconds}{}",
        cookie_name(s, name),
        if s.0.config.secure_cookie { "; Secure" } else { "" }
    );
    response
        .headers_mut()
        .append(header::SET_COOKIE, HeaderValue::from_str(&value).map_err(|_| ApiError::unavailable())?);
    Ok(())
}
fn name(value: &str) -> Result<String> {
    let value = value.trim().to_ascii_lowercase();
    if !(3..=48).contains(&value.len())
        || !value.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
    {
        return Err(ApiError::invalid("Account name must contain 3..48 lowercase letters, digits or hyphens."));
    }
    Ok(value)
}
fn label(value: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() || value.len() > 120 || value.chars().any(char::is_control) {
        return Err(ApiError::invalid("Name must contain 1..120 bytes without control characters."));
    }
    Ok(value.into())
}
fn secret(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn account(s: &Service, h: &HeaderMap) -> Result<String> {
    if h.contains_key(header::AUTHORIZATION) {
        return Err(forbidden());
    }
    let token = raw_cookie(h, &cookie_name(s, ACCOUNT_COOKIE)).filter(|v| secret(v)).ok_or_else(denied)?;
    let db = accounts(s)?.db.lock().map_err(|_| ApiError::unavailable())?;
    sql(db.query_row("SELECT a.id FROM sessions s JOIN accounts a ON a.id=s.account_id WHERE s.token_hash=?1 AND s.expires_at>?2 AND a.state='active'",params![hash(&token),now_ms()],|r|r.get(0)).optional())?.ok_or_else(denied)
}
fn recent(s: &Service, h: &HeaderMap) -> Result<String> {
    let who = account(s, h)?;
    let token = raw_cookie(h, &cookie_name(s, ACCOUNT_COOKIE)).ok_or_else(denied)?;
    let db = accounts(s)?.db.lock().map_err(|_| ApiError::unavailable())?;
    let created: i64 =
        sql(db.query_row("SELECT created_at FROM sessions WHERE token_hash=?1", [hash(&token)], |r| r.get(0)))?;
    if now_ms() - created > 600_000 {
        return Err(ApiError(
            StatusCode::FORBIDDEN,
            "fresh_sign_in_required",
            "Sign in again before changing account security or deleting your account.",
        ));
    }
    Ok(who)
}
fn rate_limit(s: &Service, h: &HeaderMap) -> Result<()> {
    // Caddy appends/sets the actual peer last. The service has no public socket.
    let peer = h
        .get("x-forwarded-for")
        .and_then(|x| x.to_str().ok())
        .and_then(|x| x.rsplit(',').next())
        .unwrap_or("local")
        .trim();
    let bucket = hash(peer);
    let window = now_ms() / 60_000;
    let db = accounts(s)?.db.lock().map_err(|_| ApiError::unavailable())?;
    sql(db.execute("DELETE FROM auth_limits WHERE window<?1", [window - 2]))?;
    let n: i64 = sql(db.query_row("SELECT count(*) FROM auth_limits", [], |r| r.get(0)))?;
    if n > 1000 {
        return Err(ApiError(
            StatusCode::TOO_MANY_REQUESTS,
            "auth_busy",
            "Too many sign-in attempts. Try again later.",
        ));
    }
    sql(db.execute(
        "INSERT INTO auth_limits VALUES(?1,?2,1) ON CONFLICT(bucket,window) DO UPDATE SET used=used+1",
        params![bucket, window],
    ))?;
    let used: i64 = sql(db.query_row(
        "SELECT used FROM auth_limits WHERE bucket=?1 AND window=?2",
        params![bucket, window],
        |r| r.get(0),
    ))?;
    if used > 20 {
        return Err(ApiError(
            StatusCode::TOO_MANY_REQUESTS,
            "auth_rate_limit",
            "Too many sign-in attempts. Try again in a minute.",
        ));
    }
    Ok(())
}
impl Accounts {
    pub(super) fn open(config: &Config) -> std::result::Result<Self, String> {
        let root = config.data_dir.join("durable");
        private_dir(&root).map_err(|_| "durable directory failed")?;
        private_dir(&root.join("projects")).map_err(|_| "project directory failed")?;
        let origin = Url::parse(&config.public_origin).map_err(|_| "invalid account origin")?;
        let rp = origin.host_str().ok_or("missing account RP host")?;
        let webauthn = WebauthnBuilder::new(rp, &origin)
            .map_err(|_| "invalid passkey RP configuration")?
            .rp_name("Memory Licorice")
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|_| "passkey initialization failed")?;
        let db = Connection::open(root.join("accounts.sqlite")).map_err(|_| "account registry failed")?;
        db.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS accounts(id TEXT PRIMARY KEY,name TEXT UNIQUE NOT NULL,uuid TEXT NOT NULL,recovery_hash TEXT,state TEXT NOT NULL,created_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS passkeys(id TEXT PRIMARY KEY,account_id TEXT NOT NULL,credential TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY,account_id TEXT NOT NULL,created_at INTEGER NOT NULL,expires_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,account_id TEXT NOT NULL,name TEXT NOT NULL,created_at INTEGER NOT NULL,delete_at INTEGER,state TEXT NOT NULL,request_id TEXT NOT NULL,digest TEXT NOT NULL,UNIQUE(account_id,request_id));
        CREATE TABLE IF NOT EXISTS project_keys(id TEXT PRIMARY KEY,project_id TEXT NOT NULL,name TEXT NOT NULL,key_hash TEXT UNIQUE NOT NULL,permission TEXT NOT NULL,created_at INTEGER NOT NULL,expires_at INTEGER,revoked INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS auth_limits(bucket TEXT NOT NULL,window INTEGER NOT NULL,used INTEGER NOT NULL,PRIMARY KEY(bucket,window));
        UPDATE projects SET state='deleted' WHERE state='creating';").map_err(|_|"account schema failed")?;
        Ok(Self { db: StdMutex::new(db), webauthn, flows: StdMutex::new(HashMap::new()), gate: Mutex::new(()) })
    }
    fn put_flow(&self, flow: Flow) -> Result<String> {
        let token = id()?;
        let mut flows = self.flows.lock().map_err(|_| ApiError::unavailable())?;
        flows.retain(|_, (deadline, _)| *deadline > now_ms());
        if flows.len() >= 64 {
            return Err(ApiError(StatusCode::TOO_MANY_REQUESTS, "auth_busy", "Too many active sign-in requests."));
        }
        flows.insert(hash(&token), (now_ms() + FLOW_MS, flow));
        Ok(token)
    }
    fn take_flow(&self, s: &Service, h: &HeaderMap) -> Result<Flow> {
        let raw = raw_cookie(h, &cookie_name(s, FLOW_COOKIE)).filter(|x| secret(x)).ok_or_else(auth_error)?;
        let (deadline, flow) =
            self.flows.lock().map_err(|_| ApiError::unavailable())?.remove(&hash(&raw)).ok_or_else(auth_error)?;
        if deadline <= now_ms() {
            return Err(auth_error());
        }
        Ok(flow)
    }
}
fn credentials(s: &Service, who: &str) -> Result<Vec<Passkey>> {
    let db = accounts(s)?.db.lock().map_err(|_| ApiError::unavailable())?;
    let mut statement = sql(db.prepare("SELECT credential FROM passkeys WHERE account_id=?1"))?;
    let rows = sql(statement.query_map([who], |r| r.get::<_, String>(0)))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| ApiError::unavailable())?;
    rows.into_iter().map(|x| serde_json::from_str(&x).map_err(|_| ApiError::unavailable())).collect()
}
fn issue_session(s: &Service, who: &str, response: Value) -> Result<Response> {
    let raw = id()?;
    let now = now_ms();
    let db = accounts(s)?.db.lock().map_err(|_| ApiError::unavailable())?;
    sql(db.execute("DELETE FROM sessions WHERE expires_at<=?1", [now]))?;
    sql(db.execute("DELETE FROM sessions WHERE token_hash IN(SELECT token_hash FROM sessions WHERE account_id=?1 ORDER BY created_at DESC LIMIT -1 OFFSET 19)",[who]))?;
    sql(db.execute("INSERT INTO sessions VALUES(?1,?2,?3,?4)", params![hash(&raw), who, now, now + SESSION_MS]))?;
    let mut response = Json(response).into_response();
    set_cookie(s, &mut response, ACCOUNT_COOKIE, &raw, SESSION_MS / 1000)?;
    set_cookie(s, &mut response, FLOW_COOKIE, "", 0)?;
    Ok(response)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NameInput {
    username: String,
}
async fn registration_start(
    State(s): State<Service>,
    h: HeaderMap,
    input: std::result::Result<Json<NameInput>, JsonRejection>,
) -> Result<Response> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    rate_limit(&s, &h)?;
    let name = name(&input.username)?;
    {
        let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        let count: i64 = sql(db.query_row("SELECT count(*) FROM accounts", [], |r| r.get(0)))?;
        if count >= MAX_ACCOUNTS {
            return Err(ApiError::quota());
        }
        let exists: bool = sql(db.query_row(
            "SELECT EXISTS(SELECT 1 FROM accounts WHERE name=?1 OR name=?2)",
            params![name, format!("deleted:{}", hash(&name))],
            |r| r.get(0),
        ))?;
        if exists {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "account_name_unavailable",
                "Choose a different account name or sign in.",
            ));
        }
    }
    let account = id()?;
    let uuid = Uuid::new_v4();
    let (challenge, state) =
        a.webauthn.start_passkey_registration(uuid, &name, &name, None).map_err(|_| auth_error())?;
    let token = a.put_flow(Flow::Register { name, account, uuid, state, existing: false })?;
    let mut r = Json(json!(challenge)).into_response();
    set_cookie(&s, &mut r, FLOW_COOKIE, &token, FLOW_MS / 1000)?;
    Ok(r)
}
async fn registration_finish(
    State(s): State<Service>,
    h: HeaderMap,
    input: std::result::Result<Json<RegisterPublicKeyCredential>, JsonRejection>,
) -> Result<Response> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    rate_limit(&s, &h)?;
    let Flow::Register { name, account: who, uuid, state, existing } = a.take_flow(&s, &h)? else {
        return Err(auth_error());
    };
    if existing && account(&s, &h)? != who {
        return Err(auth_error());
    }
    let passkey = a.webauthn.finish_passkey_registration(&input, &state).map_err(|_| auth_error())?;
    let recovery = if existing { None } else { Some(id()?) };
    {
        let mut db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        let tx = sql(db.transaction())?;
        if !existing {
            let count: i64 = sql(tx.query_row("SELECT count(*) FROM accounts", [], |r| r.get(0)))?;
            if count >= MAX_ACCOUNTS {
                return Err(ApiError::quota());
            }
            let reserved: bool = sql(tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM accounts WHERE name=?1 OR name=?2)",
                params![name, format!("deleted:{}", hash(&name))],
                |r| r.get(0),
            ))?;
            if reserved {
                return Err(ApiError(
                    StatusCode::CONFLICT,
                    "account_name_unavailable",
                    "Choose a different account name or sign in.",
                ));
            }
            sql(tx.execute(
                "INSERT INTO accounts VALUES(?1,?2,?3,?4,'active',?5)",
                params![who, name, uuid.to_string(), recovery.as_ref().map(|x| hash(x)), now_ms()],
            ))
            .map_err(|_| {
                ApiError(
                    StatusCode::CONFLICT,
                    "account_name_unavailable",
                    "Choose a different account name or sign in.",
                )
            })?;
        }
        let count: i64 = sql(tx.query_row("SELECT count(*) FROM passkeys WHERE account_id=?1", [&who], |r| r.get(0)))?;
        if count >= 5 {
            return Err(ApiError::quota());
        }
        sql(tx.execute(
            "INSERT INTO passkeys VALUES(?1,?2,?3)",
            params![
                serde_json::to_string(passkey.cred_id()).map_err(|_| ApiError::unavailable())?,
                who,
                serde_json::to_string(&passkey).map_err(|_| ApiError::unavailable())?
            ],
        ))?;
        sql(tx.commit())?;
    }
    issue_session(
        &s,
        &who,
        json!({"signed_in":true,"username":name,"recovery_key":recovery,"recovery_key_shown_once":recovery.is_some()}),
    )
}
async fn login_start(
    State(s): State<Service>,
    h: HeaderMap,
    input: std::result::Result<Json<NameInput>, JsonRejection>,
) -> Result<Response> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    rate_limit(&s, &h)?;
    let name = name(&input.username)?;
    let who: String = {
        let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        sql(db.query_row("SELECT id FROM accounts WHERE name=?1 AND state='active'", [name], |r| r.get(0)).optional())?
            .ok_or_else(auth_error)?
    };
    let creds = credentials(&s, &who)?;
    if creds.is_empty() {
        return Err(auth_error());
    }
    let (challenge, state) = a.webauthn.start_passkey_authentication(&creds).map_err(|_| auth_error())?;
    let token = a.put_flow(Flow::Login { account: who, state })?;
    let mut r = Json(json!(challenge)).into_response();
    set_cookie(&s, &mut r, FLOW_COOKIE, &token, FLOW_MS / 1000)?;
    Ok(r)
}
async fn login_finish(
    State(s): State<Service>,
    h: HeaderMap,
    input: std::result::Result<Json<PublicKeyCredential>, JsonRejection>,
) -> Result<Response> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    rate_limit(&s, &h)?;
    let Flow::Login { account: who, state } = a.take_flow(&s, &h)? else { return Err(auth_error()) };
    let result = a.webauthn.finish_passkey_authentication(&input, &state).map_err(|_| auth_error())?;
    let mut creds = credentials(&s, &who)?;
    let mut found = false;
    {
        let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        let active: bool =
            sql(db.query_row("SELECT EXISTS(SELECT 1 FROM accounts WHERE id=?1 AND state='active')", [&who], |r| {
                r.get(0)
            }))?;
        if !active {
            return Err(auth_error());
        }
        for credential in &mut creds {
            if credential.update_credential(&result).is_some() {
                found = true;
                sql(db.execute(
                    "UPDATE passkeys SET credential=?2 WHERE id=?1",
                    params![
                        serde_json::to_string(credential.cred_id()).map_err(|_| ApiError::unavailable())?,
                        serde_json::to_string(credential).map_err(|_| ApiError::unavailable())?
                    ],
                ))?;
            }
        }
    }
    if !found {
        return Err(auth_error());
    }
    a.flows
        .lock()
        .map_err(|_| ApiError::unavailable())?
        .retain(|_, (_, f)| !matches!(f,Flow::Login{account,..} if account==&who));
    issue_session(&s, &who, json!({"signed_in":true}))
}
async fn passkey_start(State(s): State<Service>, h: HeaderMap) -> Result<Response> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = recent(&s, &h)?;
    let (name, uuid): (String, String) = {
        let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        sql(db.query_row("SELECT name,uuid FROM accounts WHERE id=?1", [&who], |r| Ok((r.get(0)?, r.get(1)?))))?
    };
    let creds = credentials(&s, &who)?;
    if creds.len() >= 5 {
        return Err(ApiError::quota());
    }
    let excludes = creds.iter().map(|c| c.cred_id().clone()).collect();
    let (challenge, state) = a
        .webauthn
        .start_passkey_registration(
            Uuid::parse_str(&uuid).map_err(|_| ApiError::unavailable())?,
            &name,
            &name,
            Some(excludes),
        )
        .map_err(|_| auth_error())?;
    let token = a.put_flow(Flow::Register {
        name,
        account: who,
        uuid: Uuid::parse_str(&uuid).map_err(|_| ApiError::unavailable())?,
        state,
        existing: true,
    })?;
    let mut response = Json(json!(challenge)).into_response();
    set_cookie(&s, &mut response, FLOW_COOKIE, &token, FLOW_MS / 1000)?;
    Ok(response)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryInput {
    username: String,
    recovery_key: String,
}
async fn recover(
    State(s): State<Service>,
    h: HeaderMap,
    input: std::result::Result<Json<RecoveryInput>, JsonRejection>,
) -> Result<Response> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    rate_limit(&s, &h)?;
    let username = name(&input.username)?;
    if !secret(&input.recovery_key) {
        return Err(auth_error());
    }
    let replacement = id()?;
    let who: String;
    {
        let mut db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        let tx = sql(db.transaction())?;
        who = sql(tx
            .query_row(
                "SELECT id FROM accounts WHERE name=?1 AND recovery_hash=?2 AND state='active'",
                params![username, hash(&input.recovery_key)],
                |r| r.get(0),
            )
            .optional())?
        .ok_or_else(auth_error)?;
        sql(tx.execute("UPDATE accounts SET recovery_hash=?2 WHERE id=?1", params![who, hash(&replacement)]))?;
        sql(tx.execute("DELETE FROM sessions WHERE account_id=?1", [&who]))?;
        sql(tx.execute("DELETE FROM passkeys WHERE account_id=?1", [&who]))?;
        sql(tx.execute(
            "UPDATE project_keys SET revoked=1 WHERE project_id IN(SELECT id FROM projects WHERE account_id=?1)",
            [&who],
        ))?;
        sql(tx.commit())?;
    }
    // In-flight WebAuthn states for this account must not survive recovery.
    a.flows.lock().map_err(|_| ApiError::unavailable())?.retain(|_, (_, f)| match f {
        Flow::Login { account, .. } | Flow::Register { account, .. } => account != &who,
    });
    issue_session(
        &s,
        &who,
        json!({"signed_in":true,"recovery_key":replacement,"all_sessions_passkeys_and_api_keys_revoked":true,"add_new_passkey":true}),
    )
}
async fn rotate_recovery(State(s): State<Service>, h: HeaderMap) -> Result<Json<Value>> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = recent(&s, &h)?;
    let raw = id()?;
    let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
    sql(db.execute("UPDATE accounts SET recovery_hash=?2 WHERE id=?1", params![who, hash(&raw)]))?;
    Ok(Json(json!({"recovery_key":raw,"previous_recovery_key_revoked":true})))
}
async fn logout(State(s): State<Service>, h: HeaderMap) -> Result<Response> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let _ = account(&s, &h)?;
    let raw = raw_cookie(&h, &cookie_name(&s, ACCOUNT_COOKIE)).ok_or_else(denied)?;
    sql(a
        .db
        .lock()
        .map_err(|_| ApiError::unavailable())?
        .execute("DELETE FROM sessions WHERE token_hash=?1", [hash(&raw)]))?;
    let mut response = Json(json!({"signed_out":true})).into_response();
    set_cookie(&s, &mut response, ACCOUNT_COOKIE, "", 0)?;
    Ok(response)
}

#[derive(Clone)]
struct Project {
    id: String,
    owner: String,
    name: String,
    created: i64,
    delete_at: Option<i64>,
}
impl Project {
    fn value(&self) -> Value {
        json!({"project_id":self.id,"name":self.name,"created_at":self.created,"delete_at":self.delete_at,"retention":if self.delete_at.is_some(){"scheduled_whole_project_deletion"}else{"until_deleted"},"limits":{"observations":40,"writes":120,"source_bytes":4000,"context_tokens":2000}})
    }
}
fn project(s: &Service, project: &str, owner: Option<&str>) -> Result<Project> {
    if !secret(project) {
        return Err(missing());
    }
    let db = accounts(s)?.db.lock().map_err(|_| ApiError::unavailable())?;
    let p:Option<Project>=sql(db.query_row("SELECT p.id,p.account_id,p.name,p.created_at,p.delete_at FROM projects p JOIN accounts a ON a.id=p.account_id WHERE p.id=?1 AND p.state='active' AND a.state='active' AND(p.delete_at IS NULL OR p.delete_at>?2)",params![project,now_ms()],|r|Ok(Project{id:r.get(0)?,owner:r.get(1)?,name:r.get(2)?,created:r.get(3)?,delete_at:r.get(4)?})).optional())?;
    p.filter(|p| owner.is_none_or(|o| o == p.owner)).ok_or_else(missing)
}
#[derive(Clone)]
struct Access {
    project: Project,
    api_key_id: Option<String>,
}
fn key_access(s: &Service, h: &HeaderMap) -> Result<(String, String, String)> {
    let raw = h
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .filter(|v| secret(v))
        .ok_or_else(denied)?;
    let db = accounts(s)?.db.lock().map_err(|_| ApiError::unavailable())?;
    sql(db.query_row("SELECT k.project_id,k.permission,k.id FROM project_keys k JOIN projects p ON p.id=k.project_id JOIN accounts a ON a.id=p.account_id WHERE k.key_hash=?1 AND k.revoked=0 AND(k.expires_at IS NULL OR k.expires_at>?2) AND p.state='active' AND(p.delete_at IS NULL OR p.delete_at>?2) AND a.state='active'",params![hash(raw),now_ms()],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional())?.ok_or_else(denied)
}
pub(super) fn valid_bearer(s: &Service, h: &HeaderMap) -> bool {
    s.0.accounts.is_some() && key_access(s, h).is_ok()
}
fn access(s: &Service, h: &HeaderMap, p: &str, write: bool) -> Result<Access> {
    if h.contains_key(header::AUTHORIZATION) {
        let (id, permission, key) = key_access(s, h)?;
        if id != p || (write && permission != "write") {
            return Err(forbidden());
        }
        Ok(Access { project: project(s, p, None)?, api_key_id: Some(key) })
    } else {
        let who = account(s, h)?;
        Ok(Access { project: project(s, p, Some(&who))?, api_key_id: None })
    }
}
fn project_path(s: &Service, id: &str) -> PathBuf {
    s.0.config.data_dir.join("durable/projects").join(id)
}
async fn project_store(s: &Service, p: &Project) -> Result<(Store, Memory)> {
    let mut c = StoreConfig::new(1);
    c.readers = 1;
    let store = native(Store::open(project_path(s, &p.id).join("memory.sqlite"), c).await)?;
    let memory = native(Memory::attach(&store, &format!("project-{}", p.id)).await)?;
    Ok((store, memory))
}
fn deletion_date(days: Option<u16>) -> Result<Option<i64>> {
    match days {
        Some(d) if !(1..=365).contains(&d) => Err(ApiError::invalid(
            "retention_days must be null (until deleted) or 1..365; the entire project is deleted at its scheduled deadline.",
        )),
        Some(d) => Ok(Some(now_ms() + i64::from(d) * 86_400_000)),
        None => Ok(None),
    }
}
fn list(s: &Service, owner: &str) -> Result<Vec<Value>> {
    let db = accounts(s)?.db.lock().map_err(|_| ApiError::unavailable())?;
    let mut q=sql(db.prepare("SELECT id,account_id,name,created_at,delete_at FROM projects WHERE account_id=?1 AND state='active' AND(delete_at IS NULL OR delete_at>?2) ORDER BY created_at,id"))?;
    sql(q.query_map(params![owner, now_ms()], |r| {
        Ok(Project { id: r.get(0)?, owner: r.get(1)?, name: r.get(2)?, created: r.get(3)?, delete_at: r.get(4)? }
            .value())
    }))?
    .collect::<std::result::Result<Vec<_>, _>>()
    .map_err(|_| ApiError::unavailable())
}
async fn account_state(State(s): State<Service>, h: HeaderMap) -> Result<Json<Value>> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = account(&s, &h)?;
    let (name, created, passkeys): (String, i64, i64) = {
        let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        sql(db.query_row(
            "SELECT name,created_at,(SELECT count(*) FROM passkeys WHERE account_id=a.id) FROM accounts a WHERE id=?1",
            [&who],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        ))?
    };
    Ok(Json(
        json!({"account":{"account_id":who,"username":name,"created_at":created,"passkeys":passkeys},"projects":list(&s,&who)?,"limits":{"projects":MAX_PROJECTS,"context_tokens":BUDGET},"retrieval":s.retrieval_profile()}),
    ))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectInput {
    request_id: String,
    name: String,
    retention_days: Option<u16>,
}
async fn create_project(
    State(s): State<Service>,
    h: HeaderMap,
    input: std::result::Result<Json<ProjectInput>, JsonRejection>,
) -> Result<Json<Value>> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = account(&s, &h)?;
    check_id(&input.request_id)?;
    let label = label(&input.name)?;
    let delete_at = deletion_date(input.retention_days)?;
    let digest = hash(&json!([label, input.retention_days]).to_string());
    {
        let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        let prior: Option<(String, String, String)> = sql(db
            .query_row(
                "SELECT id,digest,state FROM projects WHERE account_id=?1 AND request_id=?2",
                params![who, input.request_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional())?;
        if let Some((id, old, state)) = prior {
            if old != digest {
                return Err(ApiError(
                    StatusCode::CONFLICT,
                    "request_conflict",
                    "request_id was already used for different project settings.",
                ));
            }
            if state != "active" {
                return Err(ApiError(
                    StatusCode::CONFLICT,
                    "project_unavailable",
                    "This creation was deleted, expired or incomplete; inspect projects before creating another.",
                ));
            }
            drop(db);
            return Ok(Json(project(&s, &id, Some(&who))?.value()));
        }
        let count: i64 = sql(db.query_row(
            "SELECT count(*) FROM projects WHERE account_id=?1 AND state IN('active','creating')",
            [&who],
            |r| r.get(0),
        ))?;
        let lifetime: i64 = sql(db.query_row("SELECT count(*) FROM projects", [], |r| r.get(0)))?;
        if count >= MAX_PROJECTS || lifetime >= MAX_PROJECT_CREATIONS {
            return Err(ApiError::quota());
        }
    }
    let id = id()?;
    let created = now_ms();
    let p = Project { id: id.clone(), owner: who.clone(), name: label.clone(), created, delete_at };
    {
        let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        sql(db.execute(
            "INSERT INTO projects VALUES(?1,?2,?3,?4,?5,'creating',?6,?7)",
            params![id, who, label, created, delete_at, input.request_id, digest],
        ))?;
    }
    private_dir(&project_path(&s, &id)).map_err(|_| ApiError::unavailable())?;
    let mut c = StoreConfig::new(1);
    c.readers = 1;
    let store = native(Store::open(project_path(&s, &id).join("memory.sqlite"), c).await)?;
    let initialized = native(Memory::initialize(&store, &format!("project-{id}")).await);
    if let Err(e) = initialized {
        native(store.close().await)?;
        return Err(e);
    }
    let conversation = random_id().map_err(|_| ApiError::unavailable())?;
    let result=native(store.write(move |w|{w.conn().execute_batch("CREATE TABLE trial_meta(name TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE trial_requests(request_id TEXT PRIMARY KEY,digest TEXT NOT NULL,status TEXT NOT NULL,response TEXT); INSERT INTO trial_meta VALUES('observations','0');")?;w.conn().execute("INSERT INTO trial_meta VALUES('conversation',?1)",[conversation])?;Ok(())}).await);
    native(store.close().await)?;
    result?;
    sql(a
        .db
        .lock()
        .map_err(|_| ApiError::unavailable())?
        .execute("UPDATE projects SET state='active' WHERE id=?1", [&id]))?;
    Ok(Json(p.value()))
}
// Tombstones are committed before removing files, so errors never restore access.
fn tombstone(s: &Service, id: &str) -> Result<()> {
    let mut db = accounts(s)?.db.lock().map_err(|_| ApiError::unavailable())?;
    let tx = sql(db.transaction())?;
    sql(tx.execute("UPDATE projects SET state='deleted',name='[deleted]' WHERE id=?1", [id]))?;
    sql(tx.execute("UPDATE project_keys SET revoked=1,name='[revoked]' WHERE project_id=?1", [id]))?;
    sql(tx.commit())?;
    Ok(())
}
fn purge(s: &Service, id: &str) -> bool {
    let path = project_path(s, id);
    !path.exists() || std::fs::remove_dir_all(path).is_ok()
}
pub(super) async fn cleanup(s: &Service) -> Result<usize> {
    let Some(a) = &s.0.accounts else { return Ok(0) };
    let _guard = a.gate.lock().await;
    let expired: Vec<String> = {
        let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        sql(db.execute("DELETE FROM sessions WHERE expires_at<=?1", [now_ms()]))?;
        let mut stmt = sql(
            db.prepare("SELECT id FROM projects WHERE state!='active' OR(delete_at IS NOT NULL AND delete_at<=?1)")
        )?;
        sql(stmt.query_map([now_ms()], |r| r.get(0)))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| ApiError::unavailable())?
    };
    for id in &expired {
        tombstone(s, id)?;
        purge(s, id);
    }
    Ok(expired.len())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RetentionInput {
    retention_days: Option<u16>,
}
async fn retention(
    State(s): State<Service>,
    h: HeaderMap,
    RoutePath(id): RoutePath<String>,
    input: std::result::Result<Json<RetentionInput>, JsonRejection>,
) -> Result<Json<Value>> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = account(&s, &h)?;
    project(&s, &id, Some(&who))?;
    let deadline = deletion_date(input.retention_days)?;
    sql(a
        .db
        .lock()
        .map_err(|_| ApiError::unavailable())?
        .execute("UPDATE projects SET delete_at=?2 WHERE id=?1", params![id, deadline]))?;
    Ok(Json(project(&s, &id, Some(&who))?.value()))
}
async fn delete_project(
    State(s): State<Service>,
    h: HeaderMap,
    RoutePath(id): RoutePath<String>,
) -> Result<Json<Value>> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = account(&s, &h)?;
    project(&s, &id, Some(&who))?;
    tombstone(&s, &id)?;
    let purged = purge(&s, &id);
    Ok(Json(json!({"deleted":true,"purge_pending":!purged})))
}
async fn delete_account(State(s): State<Service>, h: HeaderMap) -> Result<Response> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = recent(&s, &h)?;
    let projects: Vec<String> = {
        let mut db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        let tx = sql(db.transaction())?;
        let rows = {
            let mut q = sql(tx.prepare("SELECT id FROM projects WHERE account_id=?1"))?;
            sql(q.query_map([&who], |r| r.get(0)))?
                .collect::<std::result::Result<Vec<String>, _>>()
                .map_err(|_| ApiError::unavailable())?
        };
        let username: String = sql(tx.query_row("SELECT name FROM accounts WHERE id=?1", [&who], |r| r.get(0)))?;
        sql(tx.execute(
            "UPDATE accounts SET state='deleted',recovery_hash=NULL,name=?2 WHERE id=?1",
            params![who, format!("deleted:{}", hash(&username))],
        ))?;
        sql(tx.execute("DELETE FROM sessions WHERE account_id=?1", [&who]))?;
        sql(tx.execute("DELETE FROM passkeys WHERE account_id=?1", [&who]))?;
        sql(tx.execute(
            "UPDATE project_keys SET revoked=1 WHERE project_id IN(SELECT id FROM projects WHERE account_id=?1)",
            [&who],
        ))?;
        sql(tx.execute("UPDATE projects SET state='deleted',name='[deleted]' WHERE account_id=?1", [&who]))?;
        sql(tx.commit())?;
        rows
    };
    a.flows.lock().map_err(|_| ApiError::unavailable())?.retain(|_, (_, f)| match f {
        Flow::Login { account, .. } | Flow::Register { account, .. } => account != &who,
    });
    let mut purged = true;
    for id in &projects {
        purged &= purge(&s, id);
    }
    let mut response = Json(json!({"deleted":true,"purge_pending":!purged})).into_response();
    set_cookie(&s, &mut response, ACCOUNT_COOKIE, "", 0)?;
    Ok(response)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyInput {
    name: String,
    permission: String,
    expires_in_days: Option<u16>,
}
async fn keys(State(s): State<Service>, h: HeaderMap, RoutePath(id): RoutePath<String>) -> Result<Json<Value>> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = account(&s, &h)?;
    project(&s, &id, Some(&who))?;
    let rows = {
        let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        let mut stmt=sql(db.prepare("SELECT id,name,permission,created_at,expires_at,revoked FROM project_keys WHERE project_id=?1 ORDER BY created_at,id"))?;
        sql(stmt.query_map([&id],|r|Ok(json!({"key_id":r.get::<_,String>(0)?,"name":r.get::<_,String>(1)?,"permission":r.get::<_,String>(2)?,"created_at":r.get::<_,i64>(3)?,"expires_at":r.get::<_,Option<i64>>(4)?,"revoked":r.get::<_,bool>(5)?}))))?.collect::<std::result::Result<Vec<_>,_>>().map_err(|_|ApiError::unavailable())?
    };
    Ok(Json(json!({"keys":rows})))
}
async fn create_key(
    State(s): State<Service>,
    h: HeaderMap,
    RoutePath(id): RoutePath<String>,
    input: std::result::Result<Json<KeyInput>, JsonRejection>,
) -> Result<Json<Value>> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = account(&s, &h)?;
    project(&s, &id, Some(&who))?;
    let label = label(&input.name)?;
    if !matches!(input.permission.as_str(), "read" | "write") {
        return Err(ApiError::invalid("permission must be read or write (write includes read)."));
    }
    let expiry = deletion_date(input.expires_in_days)?;
    let raw = random_id().map_err(|_| ApiError::unavailable())?;
    let key_id = random_id().map_err(|_| ApiError::unavailable())?;
    {
        let db = a.db.lock().map_err(|_| ApiError::unavailable())?;
        let(count,active):(i64,i64)=sql(db.query_row("SELECT count(*),COALESCE(sum(CASE WHEN revoked=0 THEN 1 ELSE 0 END),0) FROM project_keys WHERE project_id=?1",[&id],|r|Ok((r.get(0)?,r.get(1)?))))?;
        if count >= 100 || active >= 20 {
            return Err(ApiError::quota());
        }
        sql(db.execute(
            "INSERT INTO project_keys VALUES(?1,?2,?3,?4,?5,?6,?7,0)",
            params![key_id, id, label, hash(&raw), input.permission, now_ms(), expiry],
        ))?;
    }
    Ok(Json(json!({"key_id":key_id,"api_key":raw,"permission":input.permission,"expires_at":expiry,"shown_once":true})))
}
async fn revoke_key(
    State(s): State<Service>,
    h: HeaderMap,
    RoutePath((id, key)): RoutePath<(String, String)>,
) -> Result<Json<Value>> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = account(&s, &h)?;
    project(&s, &id, Some(&who))?;
    let changed = sql(a
        .db
        .lock()
        .map_err(|_| ApiError::unavailable())?
        .execute("UPDATE project_keys SET revoked=1 WHERE id=?1 AND project_id=?2", params![key, id]))?;
    if changed == 0 {
        return Err(missing());
    }
    Ok(Json(json!({"revoked":true})))
}
async fn project_state(s: &Service, p: &Project, store: &Store, memory: &Memory) -> Result<Value> {
    Ok(
        json!({"project":p.value(),"conversation_id":current_conversation(store).await?,"memory":native(memory.snapshot_with_episodes(None).await)?.0,"supersessions":supersessions(memory).await?,"retrieval":s.retrieval_profile()}),
    )
}
async fn inspect_project(
    State(s): State<Service>,
    h: HeaderMap,
    RoutePath(id): RoutePath<String>,
) -> Result<Json<Value>> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let p = access(&s, &h, &id, false)?.project;
    let (store, memory) = project_store(&s, &p).await?;
    let result = project_state(&s, &p, &store, &memory).await;
    native(store.close().await)?;
    Ok(Json(result?))
}
async fn project_add(
    s: &Service,
    access: &Access,
    store: &Store,
    memory: &Memory,
    input: ObservationInput,
) -> Result<Value> {
    check_text(&input.text)?;
    let payload = json!({"text":input.text,"supersedes":input.supersedes}).to_string();
    if let Some(response) = reserve(store, &input.request_id, "observation", &payload).await? {
        return Ok(response);
    }
    let source_id = format!("report:{}", input.request_id);
    let snapshot = native(memory.snapshot_with_episodes(None).await)?.0;
    if let Some(stored) = snapshot.observations.iter().find(|o| o.source_id == source_id) {
        let metadata = match &stored.observation.source {
            Source::Reported { reference: Some(v) } => {
                serde_json::from_str::<Value>(v).map_err(|_| ApiError::unavailable())?
            }
            _ => return Err(ApiError::unavailable()),
        };
        if stored.observation.text != input.text || metadata["supersedes"].as_str() != input.supersedes.as_deref() {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "request_conflict",
                "request_id already names different stored evidence.",
            ));
        }
        return finish(store,&input.request_id,json!({"request_id":input.request_id,"source_id":source_id,"memory_revision":stored.recorded_revision,"supersedes":input.supersedes})).await;
    }
    if let Some(old) = &input.supersedes
        && (!snapshot.observations.iter().any(|o| &o.source_id == old)
            || source_supersessions(&snapshot).iter().any(|(s, _)| s == old))
    {
        return Err(ApiError::invalid("supersedes must name an active source in this project."));
    }
    if snapshot.observations.len() >= s.0.config.max_observations {
        return Err(ApiError::quota());
    }
    let session = current_conversation(store).await?;
    let api = access.api_key_id.is_some();
    let principal =
        access.api_key_id.as_ref().map_or_else(|| access.project.owner.clone(), |key| format!("project-key-{key}"));
    let revision=native(memory.apply(Actor{principal,role:if api{ActorRole::Importer}else{ActorRole::Human},session:session.clone()},being::Request{request_id:source_id.clone(),expected_revision:None,operation:Operation::Observe(Observation{origin:if api{Origin::ImportedDocument}else{Origin::HumanReport},subject:if api{"project-api-reporter"}else{"account-reporter"}.into(),text:input.text,relationship_id:None,occurred_at:None,source:Source::Reported{reference:Some(json!({"session":session,"supersedes":input.supersedes,"source_kind":if api{"api_report"}else{"account_report"},"key_id":access.api_key_id}).to_string())}})}).await)?.revision;
    finish(store,&input.request_id,json!({"request_id":input.request_id,"memory_revision":revision,"source_id":source_id,"supersedes":input.supersedes})).await
}
async fn add(
    State(s): State<Service>,
    h: HeaderMap,
    RoutePath(id): RoutePath<String>,
    input: std::result::Result<Json<ObservationInput>, JsonRejection>,
) -> Result<Json<Value>> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let access = access(&s, &h, &id, true)?;
    let (store, memory) = project_store(&s, &access.project).await?;
    let result = project_add(&s, &access, &store, &memory, input).await;
    native(store.close().await)?;
    Ok(Json(result?))
}
async fn search(
    State(s): State<Service>,
    h: HeaderMap,
    RoutePath(id): RoutePath<String>,
    input: std::result::Result<Json<RecallInput>, JsonRejection>,
) -> Result<Json<Value>> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let p = access(&s, &h, &id, false)?.project;
    let (store, memory) = project_store(&s, &p).await?;
    let result = packet(&s, &memory, &input.query).await;
    native(store.close().await)?;
    Ok(Json(result?))
}
async fn session(
    State(s): State<Service>,
    h: HeaderMap,
    RoutePath(id): RoutePath<String>,
    input: std::result::Result<Json<RequestId>, JsonRejection>,
) -> Result<Json<Value>> {
    let input = body(input)?;
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let p = access(&s, &h, &id, true)?.project;
    let (store, _) = project_store(&s, &p).await?;
    let result = async {
        if let Some(response) = reserve(&store, &input.request_id, "conversation", "").await? {
            return Ok(response);
        }
        let value = random_id().map_err(|_| ApiError::unavailable())?;
        let c = value.clone();
        native(
            store
                .write(move |w| {
                    w.conn().execute("UPDATE trial_meta SET value=?1 WHERE name='conversation'", [c])?;
                    Ok(())
                })
                .await,
        )?;
        finish(&store, &input.request_id, json!({"conversation_id":value})).await
    }
    .await;
    native(store.close().await)?;
    Ok(Json(result?))
}
async fn export_project(s: &Service, p: &Project) -> Result<Value> {
    let (store, memory) = project_store(s, p).await?;
    let result=async{Ok::<_,ApiError>(json!({"schema":"licorice-project-export/1","state":project_state(s,p,&store,&memory).await?,"history":native(memory.history().await)?}))}.await;
    native(store.close().await)?;
    result
}
async fn project_export(
    State(s): State<Service>,
    h: HeaderMap,
    RoutePath(id): RoutePath<String>,
) -> Result<Json<Value>> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let p = access(&s, &h, &id, false)?.project;
    Ok(Json(export_project(&s, &p).await?))
}
async fn account_export(State(s): State<Service>, h: HeaderMap) -> Result<Json<Value>> {
    let a = accounts(&s)?;
    let _guard = a.gate.lock().await;
    let who = account(&s, &h)?;
    let mut projects = vec![];
    for value in list(&s, &who)? {
        let id = value["project_id"].as_str().ok_or_else(ApiError::unavailable)?;
        projects.push(export_project(&s, &project(&s, id, Some(&who))?).await?);
    }
    Ok(Json(
        json!({"schema":"licorice-account-data-export/1","projects":projects,"credentials_included":false,"scope":"data-only export; restoring data never restores sessions, recovery keys or API credentials"}),
    ))
}
pub(super) fn router() -> Router<Service> {
    Router::new()
        .route("/account", get(|| async { Html(include_str!("../../../integrations/hosted-account/index.html")) }))
        .route(
            "/account.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    include_str!("../../../integrations/hosted-account/app.js"),
                )
            }),
        )
        .route(
            "/account.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
                    include_str!("../../../integrations/hosted-account/styles.css"),
                )
            }),
        )
        .route("/api/v2/auth/register/start", post(registration_start))
        .route("/api/v2/auth/register/finish", post(registration_finish))
        .route("/api/v2/auth/login/start", post(login_start))
        .route("/api/v2/auth/login/finish", post(login_finish))
        .route("/api/v2/auth/passkeys/start", post(passkey_start))
        .route("/api/v2/auth/recover", post(recover))
        .route("/api/v2/auth/recovery-key", post(rotate_recovery))
        .route("/api/v2/auth/logout", post(logout))
        .route("/api/v2/account", get(account_state).delete(delete_account))
        .route("/api/v2/account/export", get(account_export))
        .route("/api/v2/projects", post(create_project))
        .route("/api/v2/projects/{project}", get(inspect_project).delete(delete_project))
        .route("/api/v2/projects/{project}/retention", post(retention))
        .route("/api/v2/projects/{project}/keys", get(keys).post(create_key))
        .route("/api/v2/projects/{project}/keys/{key}", axum::routing::delete(revoke_key))
        .route("/api/v2/projects/{project}/observations", post(add))
        .route("/api/v2/projects/{project}/recall", post(search))
        .route("/api/v2/projects/{project}/sessions", post(session))
        .route("/api/v2/projects/{project}/export", get(project_export))
}
