import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "skills/being-memory/scripts/memory.py"
spec = importlib.util.spec_from_file_location("being_memory", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "agent.sqlite"
        self.conn = m.connect(self.path, "michelle", initialize=True)
        self.serial = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def request(self, kind, payload, request_id=None, revision=None):
        self.serial += 1
        r = {"request_id": request_id or f"event-{self.serial}", "kind": kind, "payload": payload}
        if kind in {"propose", "decide"}:
            r["expected_revision"] = m.project(self.conn)[0] if revision is None else revision
        return r

    def observe(self, origin="human_report", subject="lloyd", relationship=None, body="We learned that yesterday's confident answer was mistaken."):
        p = {"origin": origin, "subject": subject, "text": body}
        if relationship:
            p["relationship_id"] = relationship
        r = self.request("observe", p)
        m.apply(self.conn, r)
        return r["request_id"]

    def propose(self, cid="practice", sources=None, category="method", relationship=None, importance=0.8, statement="Check the evidence before making a confident claim."):
        p = {"claim_id": cid, "category": category, "subject": "michelle", "statement": statement,
             "source_ids": sources or [], "credence": 0.9, "importance": importance,
             "reason": "Consider a practice prompted by our work."}
        if relationship:
            p["relationship_id"] = relationship
        r = self.request("propose", p)
        m.apply(self.conn, r)
        return r

    def decide(self, stance="adopted", cid="practice", sources=None, credence=0.8):
        r = self.request("decide", {"claim_id": cid, "stance": stance, "credence": credence,
                                   "reason": "Reviewed the evidence and alternatives.", "source_ids": sources or []})
        return m.apply(self.conn, r)

    def test_observation_is_retained_without_endorsement(self):
        source = self.observe(origin="model_default_report", subject="michelle")
        self.assertEqual(m.project(self.conn)[2], {})
        self.assertEqual(m.history(self.conn)["events"][0]["request"]["request_id"], source)

    def test_high_credence_does_not_adopt_and_explicit_decision_survives_restart(self):
        self.propose()
        self.assertEqual(m.project(self.conn)[2]["practice"]["stance"], "considering")
        self.decide(credence=None)
        self.conn.close()
        self.conn = m.connect(self.path, "michelle")
        claim = m.project(self.conn)[2]["practice"]
        self.assertEqual(claim["stance"], "adopted")
        self.assertIsNone(claim["credence"])
        self.assertEqual(len(m.history(self.conn)["events"]), 2)

    def test_veto_retains_knowledge_and_does_not_assert_disproof(self):
        source = self.observe(origin="imported_document", subject="author-f")
        self.propose(sources=[source], category="belief")
        self.decide("declined", credence=0.7)
        _, obs, claims = m.project(self.conn)
        self.assertIn(source, obs)
        self.assertEqual(claims["practice"]["stance"], "declined")
        self.assertEqual(claims["practice"]["credence"], 0.7)

    def test_reconsideration_keeps_prior_decision_and_counterevidence(self):
        original = self.observe()
        self.propose(sources=[original])
        self.decide()
        contrary = self.observe(body="A later measurement contradicts the earlier result.", origin="tool_result", subject="michelle")
        self.decide("suspended", sources=[contrary], credence=0.3)
        history = m.history(self.conn)["events"]
        self.assertEqual(history[2]["request"]["payload"]["stance"], "adopted")
        c = m.project(self.conn)[2]["practice"]
        self.assertEqual(c["source_ids"], [original, contrary])
        self.assertEqual(c["stance"], "suspended")

    def test_same_request_replays_but_changed_body_and_stale_write_do_not_commit(self):
        request = self.propose()
        self.decide()
        self.assertTrue(m.apply(self.conn, request)["replayed"])
        changed = copy.deepcopy(request)
        changed["payload"]["statement"] = "A different proposition."
        with self.assertRaises(m.MemoryFault):
            m.apply(self.conn, changed)
        stale = self.request("decide", {"claim_id": "practice", "stance": "declined", "credence": None,
                                        "reason": "A stale review.", "source_ids": []}, revision=1)
        with self.assertRaises(m.MemoryFault):
            m.apply(self.conn, stale)
        self.assertEqual(len(m.history(self.conn)["events"]), 2)

    def test_wrong_agent_and_native_database_are_refused(self):
        with self.assertRaises(m.MemoryFault):
            m.connect(self.path, "ash")
        native = Path(self.tmp.name) / "native.sqlite"
        with sqlite3.connect(native) as c:
            c.execute("CREATE TABLE facts(id INTEGER)")
            c.execute("INSERT INTO facts VALUES(7)")
        with self.assertRaises(m.MemoryFault):
            m.connect(native, "michelle", initialize=True)
        with sqlite3.connect(native) as c:
            self.assertEqual(c.execute("SELECT id FROM facts").fetchall(), [(7,)])

    def test_imported_autobiography_cannot_become_agent_experience(self):
        source = self.observe(origin="imported_document", subject="author-f")
        with self.assertRaises(m.MemoryFault):
            self.propose(category="autobiographical", sources=[source])
        self.assertEqual(len(m.history(self.conn)["events"]), 1)
        self.propose(category="method", sources=[source])
        self.decide()
        self.assertEqual(m.project(self.conn)[2]["practice"]["category"], "method")

    def test_participation_can_be_autobiographical_but_later_import_cannot_extend_it(self):
        source = self.observe(origin="direct_interaction", subject="michelle")
        self.propose(category="autobiographical", sources=[source])
        other = self.observe(origin="imported_document", subject="author-f")
        with self.assertRaises(m.MemoryFault):
            self.decide(sources=[other])
        self.decide()

    def test_another_participants_event_cannot_be_this_agents_autobiography(self):
        source = self.observe(origin="direct_interaction", subject="other-agent")
        request = self.request("propose", {"claim_id": "other-life", "category": "autobiographical",
            "subject": "other-agent", "statement": "Another participant did this.", "source_ids": [source],
            "credence": None, "importance": 0.7, "reason": "Deliberately invalid self-attribution."})
        with self.assertRaises(m.MemoryFault):
            m.apply(self.conn, request)
        self.assertEqual(m.project(self.conn)[0], 1)

    def test_self_description_cannot_silently_describe_another_identity(self):
        request = self.request("propose", {"claim_id": "other-self", "category": "self_description",
            "subject": "other-agent", "statement": "I am this other participant.", "source_ids": [],
            "credence": None, "importance": 0.7, "reason": "Deliberately invalid identity substitution."})
        with self.assertRaises(m.MemoryFault):
            m.apply(self.conn, request)
        self.assertEqual(m.project(self.conn)[0], 0)

    def test_relationship_scope_cannot_leak_via_global_claim_or_other_recall(self):
        source = self.observe(relationship="lloyd-michelle")
        with self.assertRaises(m.MemoryFault):
            self.propose(sources=[source])
        self.propose(sources=[source], relationship="lloyd-michelle", statement="Our private working agreement.")
        self.assertNotIn("private working agreement", m.context(self.conn, "michelle")["text"])
        self.assertNotIn("private working agreement", m.context(self.conn, "michelle", "other-person")["text"])
        self.assertIn("private working agreement", m.context(self.conn, "michelle", "lloyd-michelle")["text"])

    def test_large_whole_source_is_omitted_and_small_claim_still_fits_unicode_cap(self):
        source = self.observe(body="界" * 1000)
        self.propose(cid="large", sources=[source], importance=1)
        self.propose(cid="small", statement="Keep uncertainty visible.", importance=0.5)
        packet = m.context(self.conn, "michelle")
        self.assertEqual(packet["omitted_claim_ids"], ["large"])
        self.assertIn("Keep uncertainty visible.", packet["text"])
        self.assertNotIn("界", packet["text"])
        self.assertLessEqual(len(packet["text"].encode()), 2000)

    def test_two_reviewers_cannot_overwrite_an_intervening_decision(self):
        self.propose()
        other = m.connect(self.path, "michelle")
        try:
            old_revision = m.context(other, "michelle")["revision"]
            self.decide()
            request = {"request_id": "other-review", "kind": "decide", "expected_revision": old_revision,
                       "payload": {"claim_id": "practice", "stance": "declined", "credence": None,
                                   "reason": "A concurrent review.", "source_ids": []}}
            with self.assertRaises(m.MemoryFault):
                m.apply(other, request)
            self.assertEqual(m.project(other)[2]["practice"]["stance"], "adopted")
        finally:
            other.close()

    def test_receipt_write_failure_leaves_prior_stance_and_revision(self):
        self.propose()
        self.conn.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT,'test interruption'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.decide()
        self.assertEqual(m.project(self.conn)[0], 1)
        self.assertEqual(m.project(self.conn)[2]["practice"]["stance"], "considering")

    def test_boundary_replay_never_creates_a_claim(self):
        r = self.request("boundary", {"session_id": "session-a", "turn_id": "turn-1", "phase": "completed"})
        m.apply(self.conn, r)
        self.assertTrue(m.apply(self.conn, r)["replayed"])
        self.assertEqual(m.project(self.conn)[2], {})

    def test_invalid_weights_unknown_sources_and_duplicate_json_are_rejected(self):
        for v in [True, float("nan"), float("inf"), -0.1, 1.1]:
            with self.assertRaises(m.MemoryFault):
                m.weight(v)
        with self.assertRaises(m.MemoryFault):
            self.propose(sources=["missing"])
        with self.assertRaises(m.MemoryFault):
            m.parse_json('{"kind":"observe","kind":"decide"}')
        with self.assertRaises(m.MemoryFault):
            m.parse_json('{"value":NaN}')
        self.assertEqual(m.project(self.conn)[0], 0)

    def test_same_seed_then_different_recorded_experiences_produces_distinct_portable_state(self):
        self.propose()
        self.conn.close()
        other_path = Path(self.tmp.name) / "fork.sqlite"
        other_path.write_bytes(self.path.read_bytes())
        self.conn = m.connect(self.path, "michelle")
        other = m.connect(other_path, "michelle")
        try:
            self.decide("adopted")
            m.apply(other, {"request_id": "other-choice", "kind": "decide", "expected_revision": 1,
                            "payload": {"claim_id": "practice", "stance": "suspended", "credence": None,
                                        "reason": "Different work leaves this unresolved.", "source_ids": []}})
            self.assertNotEqual(m.context(self.conn, "michelle")["text"], m.context(other, "michelle")["text"])
            self.assertEqual(m.history(other)["events"][0], m.history(self.conn)["events"][0])
        finally:
            other.close()

    def test_cli_works_from_another_directory_and_failure_is_structured(self):
        prefix = [sys.executable, str(SCRIPT), "--store", str(self.path), "--agent", "michelle"]
        result = subprocess.run(prefix + ["context"], cwd=self.tmp.name, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["ok"])
        bad = subprocess.run(prefix + ["apply"], input='{"kind":"decide"}', cwd=self.tmp.name, capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)
        self.assertEqual(bad.stderr, "")
        self.assertFalse(json.loads(bad.stdout)["ok"])

    def test_context_does_not_create_an_uninitialized_store(self):
        missing = Path(self.tmp.name) / "not-created.sqlite"
        with self.assertRaises(m.MemoryFault):
            m.connect(missing, "ash")
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
