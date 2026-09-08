"""Single place where every input path lives, so a data move is a one-line edit."""
import os

NLD = "/root/nld"
E14 = f"{NLD}/zombie-fix/outputs/e14_uncapped"          # uncapped base harness, 15 rollouts
# Both arms of the uncapped E14 grid: the base harness and the hand-engineered
# flag set (netplay_telemetry + melee_hints + tool_tier=human + skill_doc_coords).
E14_ARMS = {
    "base":  {f"base_r{r}__prime_agent": r for r in (1, 2, 3)},
    "human": {f"human_r{r}__prime_agent": r for r in (1, 2, 3)},
}
ARM_LABELS = {"base": "base harness", "human": "hand-engineered flags"}
E14_REPS = E14_ARMS["base"]          # kept so single-arm callers stay valid
SWEEPS = f"{NLD}/e14/outputs"                            # density / encoding ablation runs
# ---- human corpus selector -------------------------------------------------
# The NLD-NAO decode exists in two vintages: "v1" (28,040 games) and the "v2"
# superset (60,130 games, v1 verified as a strict subset). Set HUMAN_CORPUS=v1
# to reproduce the original figures; the default is the v2 superset.
HUMAN_CORPUS = os.environ.get("HUMAN_CORPUS", "v2")
_CSUF = {"v1": "", "v2": "_v2"}[HUMAN_CORPUS]
COMPARE = f"{NLD}/compare_data{_CSUF}.json"              # NLD human population curves
COMPARE_V1 = f"{NLD}/compare_data.json"                  # v1 kept reachable
COMPARE_V2 = f"{NLD}/compare_data_v2.json"
# The NAO top-10 corpus is the DeepMind `nao_top10` tty-tensor release, decoded
# by top10_curves.py from /root/nld/top10/*.npz. It is a DIFFERENT source from
# the NLD-NAO shard behind all_games*.json -- none of its ten players appear in
# that shard -- so it has no v1/v2 vintage and does not move with the superset.
TOP10 = f"{NLD}/top10_curves.json"                       # per-game NAO top-10 summaries
ALL_GAMES = f"{NLD}/all_games{_CSUF}.json"               # per-game NLD-NAO decode (v1/v2)
# Which population defines the experience-for-depth norm drawn in fig4:
#   "winners" -- the NLD-NAO ascension trajectories (433 on v1, 824 on v2)
#   "top10"   -- the 12,154 DeepMind nao_top10 games (the original fig4 curve)
NORM_SOURCE = os.environ.get("NORM_SOURCE", "winners")
ASCENSION = f"{NLD}/games_ascension.json"                 # 24 decoded NAO ascension games
ENDGAME = f"{NLD}/endgame_states.json"                    # plane / ascension turns per game
BALROG = f"{NLD}/e14/environments/nethack/nethack_harness/prompt/balrog_achievements.json"
FIGDATA = "/root/overleaf/figs/figdata.json"
IMAGES = "/root/overleaf/images"
