"""The stale-code detector must actually fire — otherwise it is decoration.

This is the guard against the third silent failure of the project: a forked worker
pool running a superseded estimator while the repository held the new one, producing
timings that described code no longer in the tree.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest

from src import provenance


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    """A file registered as if it were an estimation module, that we can then edit."""
    path = tmp_path / "estimator_stub.py"
    path.write_text("VERSION = 1\n")
    monkeypatch.setattr(provenance, "_AT_IMPORT", dict(provenance._AT_IMPORT))
    digest = provenance.register(str(path))
    return path, digest


def test_clean_tree_reports_no_staleness(fake_module):
    assert provenance.stale_modules() == {}
    stamp = provenance.assert_current()
    assert "estimator_stub" in stamp["modules"]
    assert stamp["stale"] == {}


def test_editing_a_registered_module_is_detected(fake_module):
    """The exact scenario: source changes after import, memory still holds the old."""
    path, digest = fake_module
    path.write_text("VERSION = 2\n")

    stale = provenance.stale_modules()
    assert "estimator_stub" in stale
    assert stale["estimator_stub"]["imported"] == digest
    assert stale["estimator_stub"]["on_disk"] != digest

    with pytest.raises(RuntimeError, match="stale code"):
        provenance.assert_current()
    provenance.assert_current(strict=False)  # must not raise


def test_hash_is_captured_at_import_not_recomputed(fake_module):
    """The subtlety that makes this work at all.

    Re-hashing the file at check time would report agreement in a forked child, since
    the child reads the *new* bytes from disk while executing the *old* bytes in
    memory. Only a value frozen at import travels with the fork, so the registered
    digest must not track later edits.
    """
    path, digest = fake_module
    path.write_text("VERSION = 3\n")
    assert provenance._AT_IMPORT["estimator_stub"][1] == digest


def _child_reports_stale(q):
    q.put(provenance.stale_modules())


@pytest.mark.skipif(os.name != "posix", reason="fork is POSIX-only")
def test_a_forked_child_inherits_the_stale_stamp(fake_module):
    """End to end: edit, fork, and the child must report itself stale."""
    path, digest = fake_module
    path.write_text("VERSION = 4\n")

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child_reports_stale, args=(q,))
    p.start()
    stale = q.get(timeout=30)
    p.join(timeout=30)

    assert "estimator_stub" in stale, "a forked child failed to notice it is stale"
    assert stale["estimator_stub"]["imported"] == digest


def test_real_estimation_modules_are_registered():
    """If a module is not registered its staleness is invisible."""
    import src.analysis.attribution  # noqa: F401
    import src.glue.core  # noqa: F401
    import src.pipeline  # noqa: F401

    assert {"core", "pipeline", "attribution"} <= set(provenance._AT_IMPORT)


def test_code_stamp_is_json_ready_and_identifies_the_commit():
    import json

    stamp = provenance.code_stamp()
    json.dumps(stamp)
    assert set(stamp) == {"git_sha", "git_dirty", "modules", "stale"}
    assert isinstance(stamp["git_dirty"], bool)
