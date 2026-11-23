"""The agreement table of docs/envs.md from the JSON reports of scripts/envs/dmc_check.py / brax_check.py."""
import json
import sys


def f(x):
    return "-" if x is None else f"{x:.1e}"


def main(paths):
    dmc, brax = [], []
    for p in paths:
        d = json.load(open(p))
        o, m = d["openloop"], d["matched"]
        if "dm_obs_dim" in d:
            q = o["qpos_max_abs_median_over_episodes"]
            b = o.get("baseline_dm_perturbed_1e-6_qpos_max_abs_median", {})
            dmc.append(f"| {d['task']} | {d['obs_dim']} | {d['refine']} | {f(max(m['obs_max_abs'].values()))} | "
                       f"{f(m['reward_max_abs'])} | {f(q['10'])} / {f(b.get('10'))} | {f(q['100'])} / {f(b.get('100'))} | "
                       f"{f(o['reward_abs_diff_mean'])} |")
        else:
            q = o["qpos_max_abs_median"]
            brax.append(f"| {d['task']} | {d['obs_dim']} | {f(m['obs_max_abs'])} | {f(m['reward_max_abs'])} | "
                        f"{m['termination_agree']:.3f} | {f(q.get('10'))} | {f(q.get('25'))} | {f(q.get('50'))} | "
                        f"{o['alive_at_end']}/{o['episodes']} |")
    out = []
    if dmc:
        out += ["DMC (dm_control 1.0.51, MuJoCo C float64; open loop: 4 episodes of uniform random actions, qpos max-abs "
                "error, median over the episodes, engine / dm_control's own from a 1e-6 perturbation):", "",
                "| task | obs | refine | matched obs | matched reward | open loop @10 | open loop @100 | reward diff (mean) |",
                "|---|---|---|---|---|---|---|---|"] + sorted(dmc) + [""]
    if brax:
        out += ["Brax (0.14.2, MJX; open loop: 16 episodes of uniform random actions until a termination, qpos max-abs "
                "error, median over the live episodes):", "",
                "| task | obs | matched obs | matched reward | termination agree | open loop @10 | @25 | @50 | alive @100 |",
                "|---|---|---|---|---|---|---|---|---|"] + sorted(brax) + [""]
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
