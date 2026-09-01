"""Reserved stream ids must not appear in any design-decision artifact.

The reserved set never informs lr0, the rank grid, or the stopping criterion
(`docs/16` §Amendments A3). Outcome files that *evaluate* the reserved set after
those decisions are already fixed are not design-decision artifacts; they are
listed as `not_design_decision` in the reservation file.
"""

from __future__ import annotations

from src.reservations import (
    RESERVATION_PATH,
    ROOT,
    design_decision_paths,
    load,
    old_init_seeds,
    reserved_ids_in_path,
    reserved_stream_ids,
)


def test_reservation_file_names_two_populations():
    rec = load()
    assert rec["rule"].startswith("The reserved set never informs")
    old = rec["populations"]["old"]
    reserved = rec["populations"]["reserved"]
    assert old["keying"] == "legacy_seed"
    assert reserved["keying"] == "stream_rng"
    assert old_init_seeds() == tuple(range(8))
    ids = reserved_stream_ids()
    assert ids == frozenset(range(10000, 10008))
    assert rec["stream_rng"]["salt"] == 20260817


def test_reserved_ids_are_outside_the_used_label_block():
    """0–7 were stored as labels and used as genuine-n keys after the RNG fix."""
    assert min(reserved_stream_ids()) >= 10000


def test_reserved_id_fails_if_it_appears_in_a_design_decision_artifact():
    reserved = reserved_stream_ids()
    hits = []
    for path in design_decision_paths():
        found = reserved_ids_in_path(path, reserved)
        if found:
            hits.append(f"{path.relative_to(ROOT)}: {sorted(found)}")
    assert hits == [], (
        "reserved stream id in a design-decision artifact — the reserved set "
        "never informs lr0, the rank grid, or the stopping criterion:\n  "
        + "\n  ".join(hits)
    )


def test_the_reservation_file_itself_is_not_a_design_decision_artifact():
    rec = load()
    listed = [p.resolve() for p in design_decision_paths(rec)]
    assert RESERVATION_PATH.resolve() not in listed
