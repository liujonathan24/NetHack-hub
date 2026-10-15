"""Core envs rebuilt from recorded key sequences (tests/golden/codehack_replays.json)."""

import json
import pathlib

GOLDEN = json.loads((pathlib.Path(__file__).parent / "golden" / "codehack_replays.json").read_text())


def core_env(seed=0, character="Val-hum-neu-fem", modify=None, keys=()):
    """A fresh core env on the fixed-mode settings, past the welcome, with `keys` already pressed."""
    from nethack_core.env import NetHackCoreEnv

    env = NetHackCoreEnv(task_name="NetHackScore-v0", max_episode_steps=100_000, tune={"reveal_map": 1.0}, modify=modify)
    env.seed(seed, seed)
    obs, _ = env.reset(character=character)
    env.setup_character = character
    while b"--More--" in obs.tty_chars[:3].tobytes():
        obs = env.step(13)[0]
    for key in keys:
        env.step(int(key))
    return env


def core_from_golden(name):
    golden = GOLDEN[name]
    return core_env(golden["seed"], golden["character"], golden["modify"], golden["keys"])
