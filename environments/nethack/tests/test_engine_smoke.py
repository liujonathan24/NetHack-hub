import os

from nethack_core import _engine


def test_start_step_end_runs():
    env = _engine.RawEngine()
    obs = env.start(core=42, disp=42)
    assert obs.tty_chars.shape == (24, 80)
    assert obs.glyphs.shape == (21, 79)
    assert obs.blstats.shape == (27,)
    obs2 = env.step(0)            # action index 0
    assert obs2.tty_chars.shape == (24, 80)
    env.end()


def test_two_instances_sequential():
    # one engine per process is fine; ensure a fresh start after end works
    env = _engine.RawEngine()
    env.start(core=1, disp=1)
    env.step(0)
    env.end()


def test_start_without_end_does_not_leak():
    """Re-entering start() without end() must not leak hackdirs.

    The engine keeps ONE writable hackdir per instance and reuses it across
    start()/reset() (a perf optimization — see RawEngine._ensure_hackdir: a
    ~28x-faster reset that avoids re-copying the data tree). So a second start()
    without end() cleans and REUSES the same directory rather than allocating a
    fresh one — which is precisely why nothing leaks. Only end() removes it.
    """
    env = _engine.RawEngine()
    env.start(core=1, disp=1)
    first_hackdir = env._hackdir

    # Re-enter start() without calling end() first: the hackdir is reused, not
    # re-allocated, so the SAME directory persists (no second, orphaned dir).
    env.start(core=2, disp=2)
    assert env._hackdir == first_hackdir, (
        "start() re-entry should reuse the same hackdir, not allocate a new one"
    )
    assert os.path.isdir(first_hackdir), (
        f"the reused hackdir should still exist between games: {first_hackdir}"
    )

    # Engine must still be functional after re-entry.
    env.step(0)

    # end() is the sole teardown that removes the (single, reused) hackdir.
    env.end()
    assert not os.path.exists(first_hackdir), (
        f"hackdir was not cleaned up after end(): {first_hackdir}"
    )
