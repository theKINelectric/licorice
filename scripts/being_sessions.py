"""Explicit identity profiles for local Codex/Claude conversation resumption.

This module never edits harness transcripts, changes their auth/history homes,
starts a model itself, or grants additional tool permissions.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
loader = SourceFileLoader("being_legacy_binding", str(ROOT / "scripts/claude-being"))
spec = spec_from_loader(loader.name, loader)
binding_api = module_from_spec(spec)
loader.exec_module(binding_api)
DEFAULT_REGISTRY = ROOT / "data/being/profiles.json"
SKILL = ROOT / "integrations/being-memory/skills/being-memory/SKILL.md"
MOD = ROOT / "integrations/being-memory"


def clean(value):
    return isinstance(value, str) and value and not any(ord(c) < 32 or ord(c) == 127 for c in value)


def absolute(value):
    if not clean(value) or not Path(value).is_absolute():
        raise ValueError("Use an absolute path without control characters")
    return Path(value)


def same_file(a, b):
    a, b = Path(a), Path(b)
    return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))


def validate_registry(value):
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Expected a version1 profile registry")
    profiles, routes = value.get("profiles"), value.get("sessions")
    if not isinstance(profiles, dict) or not isinstance(routes, dict):
        raise ValueError("Invalid profile registry structure")
    checked = []
    for name, profile in profiles.items():
        if not binding_api.ID.fullmatch(name) or not isinstance(profile, dict):
            raise ValueError("Invalid profile name")
        if profile.get("harness") not in ["codex", "claude"]:
            raise ValueError("Select codex or claude explicitly")
        binding = binding_api.validate_binding(profile.get("binding", {}))
        for field in ["store", "binary", "models"]:
            absolute(binding[field])
        absolute(profile.get("runtime"))
        if profile.get("credential_file") is not None:
            absolute(profile["credential_file"])
            if profile["harness"] != "claude":
                raise ValueError("Only Claude profiles support an explicit Anthropic credential file")
        for other in checked:
            b = other["binding"]
            if binding["agent_id"] == b["agent_id"] or binding["relationship_id"] == b["relationship_id"]:
                raise ValueError("Distinct profiles require separate agent identities and relationship scopes")
            if same_file(binding["store"], b["store"]):
                raise ValueError("Distinct profiles must not share or alias a memory file")
        checked.append(profile)
    for key, route in routes.items():
        if not clean(key) or not isinstance(route, dict) or route.get("profile") not in profiles:
            raise ValueError("Invalid saved session route")
        profile = profiles[route["profile"]]
        if key.split(":", 1)[0] != profile["harness"]:
            raise ValueError("Session route has the wrong harness")
    return value


def read_registry(path):
    if not path.exists():
        return {"version": 1, "profiles": {}, "sessions": {}}
    if path.stat().st_size > 1_000_000:
        raise ValueError("Profile registry exceeds the local size limit")
    return validate_registry(json.loads(path.read_text()))


@contextmanager
def registry_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def save_registry(path, value):
    validate_registry(value)
    fd, temporary = tempfile.mkstemp(prefix=".being-profiles-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def runtime_check(profile):
    path = absolute(profile["runtime"])
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("Profile runtime must be an existing executable")
    r = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=15)
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", r.stdout)
    if r.returncode or not match:
        raise ValueError("Cannot qualify the selected runtime version")
    minimum = (2, 1, 287) if profile["harness"] == "claude" else (0, 163, 0)
    label = "claude" if profile["harness"] == "claude" else "codex"
    if label not in r.stdout.lower() or tuple(map(int, match.groups())) < minimum:
        raise ValueError("Selected runtime does not support the qualified resume interface")
    return r.stdout.strip()


def preflight(profile):
    binding = profile["binding"]
    if not Path(binding["store"]).is_file():
        raise ValueError("Initialize the selected native memory explicitly before registration")
    if not Path(binding["models"]).is_dir():
        raise ValueError("Selected local models directory is unavailable")
    state = binding_api.native(binding, "snapshot")
    if state["agent_id"] != binding["agent_id"]:
        raise ValueError("Selected memory owner does not match the profile")
    return {"version": runtime_check(profile), "revision": state["revision"]}


def register(path, name, harness, binding_path, runtime, credential_file=None):
    if not binding_api.ID.fullmatch(name):
        raise ValueError("Invalid profile name")
    binding = binding_api.read_binding(absolute(binding_path))
    candidate = {"harness": harness, "runtime": str(absolute(runtime)), "binding": binding,
                 "credential_file": str(absolute(credential_file)) if credential_file else None}
    with registry_lock(path):
        registry = read_registry(path)
        if name in registry["profiles"]:
            if registry["profiles"][name] == candidate:
                preflight(candidate)
                return {"profile": name, "registered": True, "replayed": True}
            raise ValueError("Profile already exists with a different selection; use a distinct name")
        registry["profiles"][name] = candidate
        validate_registry(registry)
        preflight(candidate)
        save_registry(path, registry)
    return {"profile": name, "registered": True, "replayed": False}


def selector_key(harness, resume):
    if not clean(resume) or resume.startswith("-") or "/" in resume or "\\" in resume or resume.endswith(".jsonl"):
        raise ValueError("Use an explicit non-option saved session ID or name")
    try:
        resume = str(uuid.UUID(resume))
    except ValueError:
        pass  # Names are pinned as literal selectors; UUIDs identify sessions.
    return harness + ":" + resume


def check_route(registry, name, resume):
    if resume is not None:
        key = selector_key(registry["profiles"][name]["harness"], resume)
        previous = registry["sessions"].get(key)
        if previous and previous["profile"] != name:
            raise ValueError("This session selector is pinned to a different profile; unpin it explicitly first")


def attachment_note(name, profile, project):
    return (
        "Attach Licorice BEING memory to this existing conversation. Continue its saved history; "
        "do not reset it or invent participation in imported history. "
        "Use the being-memory skill at " + json.dumps(str(SKILL)) + ". "
        "The explicit project is " + json.dumps(str(project)) + ". "
        "The selected profile is " + json.dumps(name) + ". "
        "Use this complete binding as a unit, even if project-local defaults name another agent: "
        + json.dumps(profile["binding"], sort_keys=True) + ". "
        "Retrieve relevant continuity under the shared 2000-token memory/evidence cap before continuing. "
        "Record consequential corrections or outcomes deliberately, with attributed sources; "
        "adoption requires a separate decision. No automatic biography, stance adoption or history import. "
        "Existing tool permissions still apply; request normal access if this memory is outside them."
    )


def plan(path, name, project, resume=None, last=False, model=None):
    registry = read_registry(path)
    if name not in registry["profiles"]:
        raise ValueError("Unknown identity profile")
    profile = registry["profiles"][name]
    project = absolute(project).resolve()
    if not project.is_dir():
        raise ValueError("Selected project must be an existing directory")
    if (resume is None) == (not last):
        raise ValueError("Select exactly one saved session or --last")
    check_route(registry, name, resume)
    qualified = preflight(profile)
    env = binding_api.memory_environment(profile["binding"], {})
    env["LICORICE_BEING_PROFILE"] = name
    note = attachment_note(name, profile, project)
    if model is not None and (not clean(model) or model.startswith("-")):
        raise ValueError("Invalid model selection")
    if profile["harness"] == "codex":
        argv = [profile["runtime"], "resume"] + (["--last"] if last else [resume])
        argv += ["--cd", str(project), "-c", 'tui.resume_cwd="current"']
        # Pin only memory variables, preserving other shell policy/settings.
        # This also works when the TUI uses an already running app-server.
        for key, value in sorted(env.items()):
            argv += ["-c", "shell_environment_policy.set." + key + "=" + json.dumps(value, ensure_ascii=False)]
    else:
        if os.environ.get("CLAUDE_CODE_SAFE_MODE", "").lower() in ["1", "true"]:
            raise ValueError("Claude safe mode disables Mods; no memory attach launched")
        argv = [profile["runtime"], "--plugin-dir", str(MOD), "--settings", json.dumps({"env": env}, sort_keys=True)]
        argv += ["--continue"] if last else ["--resume", resume]
    if model:
        argv += ["--model", model]
    argv.append(note)
    return {"profile": name, "harness": profile["harness"], "agent_id": profile["binding"]["agent_id"],
            "store": profile["binding"]["store"], "project": str(project), "resume": resume,
            "last": last, "argv": argv, "memory_environment": env,
            "credential_file": profile.get("credential_file"), "qualified": qualified,
            "model": model, "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
            "history_policy": "Resume in place; append one visible attachment message; no fork or transcript rewrite.",
            "session_pin": "explicit selector" if resume else "not available for --last",
            "memory_budget_tokens": 2000}


def reject_running_claude(profile, project, resume):
    # A running background session can attach to the old process or copy itself,
    # so it cannot honor this launch's profile/Mod guarantees.
    r = subprocess.run([profile["runtime"], "agents", "--json", "--all"], capture_output=True, text=True, timeout=15)
    if r.returncode or len(r.stdout) > 2_000_000:
        raise ValueError("Cannot check Claude background sessions; no resume launched")
    agents = json.loads(r.stdout)
    if not isinstance(agents, list):
        raise ValueError("Unsupported Claude background-session catalog")
    terminal = {"done", "stopped", "failed", "cancelled", "canceled", "killed"}
    for agent in agents:
        if resume:
            candidates = [agent.get("id"), agent.get("sessionId"), agent.get("name")]
            match = any(selector_key("claude", resume) == selector_key("claude", value)
                        for value in candidates if clean(value) and not value.startswith("-")
                        and "/" not in value and "\\" not in value and not value.endswith(".jsonl"))
        else:
            match = bool(agent.get("cwd")) and Path(agent["cwd"]).resolve() == Path(project)
        if match and agent.get("state") not in terminal:
            raise ValueError("Stop the existing Claude background session before resuming with a memory profile")


def launch(path, prepared):
    with registry_lock(path):
        registry = read_registry(path)
        profile = registry["profiles"][prepared["profile"]]
        check_route(registry, prepared["profile"], prepared["resume"])
        current = plan(path, prepared["profile"], prepared["project"], prepared["resume"], prepared["last"], prepared["model"])
        if current["profile_sha256"] != prepared["profile_sha256"] or current["argv"] != prepared["argv"]:
            raise ValueError("Profile changed during preflight; no launch")
        if profile["harness"] == "claude":
            reject_running_claude(profile, prepared["project"], prepared["resume"])
        if prepared["resume"]:
            key = selector_key(profile["harness"], prepared["resume"])
            registry["sessions"][key] = {"profile": prepared["profile"], "project": prepared["project"]}
            save_registry(path, registry)
    env = binding_api.memory_environment(profile["binding"], os.environ)
    env["LICORICE_BEING_PROFILE"] = prepared["profile"]
    if profile.get("credential_file") and not env.get("ANTHROPIC_API_KEY"):
        key = binding_api.credential(Path(profile["credential_file"]))
        if not key:
            raise ValueError("Selected credential file has no Anthropic key; no fallback file was used")
        env["ANTHROPIC_API_KEY"] = key
    os.chdir(prepared["project"])
    os.execve(prepared["argv"][0], prepared["argv"], env)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resume local Codex/Claude history with an explicit memory profile and project.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Absolute private profile registry (default: Licorice data/being/profiles.json)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("profiles", help="List registered identities without model calls")
    add = commands.add_parser("register", help="Register an already initialized complete native binding")
    add.add_argument("name"); add.add_argument("--harness", choices=["codex", "claude"], required=True)
    add.add_argument("--binding", required=True); add.add_argument("--runtime", required=True)
    add.add_argument("--credential-file", help="Optional explicit Anthropic dotenv file for this Claude identity")
    check = commands.add_parser("check", help="Check owner and runtime locally")
    check.add_argument("--profile", required=True)
    attach = commands.add_parser("attach", help="Resume a stopped, saved local conversation")
    attach.add_argument("--profile", required=True); attach.add_argument("--project", required=True)
    select = attach.add_mutually_exclusive_group(required=True)
    select.add_argument("--resume"); select.add_argument("--last", action="store_true")
    attach.add_argument("--model"); attach.add_argument("--dry-run", action="store_true")
    unpin = commands.add_parser("unpin", help="Remove only a local selector/profile pin, never conversation history")
    unpin.add_argument("--harness", choices=["codex", "claude"], required=True); unpin.add_argument("--resume", required=True)
    args = parser.parse_args(argv)
    path = absolute(args.registry)
    if args.command == "register":
        result = register(path, args.name, args.harness, args.binding, args.runtime, args.credential_file)
    elif args.command == "profiles":
        result = {"profiles": [{"name": name, "harness": p["harness"], "agent_id": p["binding"]["agent_id"],
                                "store": p["binding"]["store"]} for name, p in sorted(read_registry(path)["profiles"].items())]}
    elif args.command == "check":
        registry = read_registry(path)
        if args.profile not in registry["profiles"]:
            raise ValueError("Unknown identity profile")
        result = {"profile": args.profile, **preflight(registry["profiles"][args.profile])}
    elif args.command == "unpin":
        with registry_lock(path):
            registry = read_registry(path)
            removed = registry["sessions"].pop(selector_key(args.harness, args.resume), None) is not None
            save_registry(path, registry)
        result = {"removed_route": removed, "conversation_history_changed": False}
    else:
        result = plan(path, args.profile, args.project, args.resume, args.last, args.model)
        if not args.dry_run:
            launch(path, result)
    print(json.dumps(result, indent=2))
