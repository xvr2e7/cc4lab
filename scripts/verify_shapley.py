"""Independent check of ``cc4lab.attribution.shapley_values`` (and the rollout plumbing under it).

    uv run python scripts/verify_shapley.py            # games only, seconds
    uv run python scripts/verify_shapley.py --rollouts # plus planted-culprit checks on real rollouts

Nothing here reuses the code under test for its expected answers:

* the oracle is a brute force over all player orderings in exact rational
  arithmetic, written on Python sets (no bit tricks);
* the textbook games below have closed-form Shapley values that can be worked
  out on paper (given in the comments).

Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from fractions import Fraction as F

import numpy as np

from cc4lab.attribution import coalition_masks
from cc4lab.attribution.loo import loo_values
from cc4lab.attribution.shapley import shapley_values

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def table(fn, n: int) -> list:
    """Coalition values in the repo's index order, built from the repo's own masks (so the convention is tested too)."""
    return [fn(frozenset(np.flatnonzero(row).tolist())) for row in coalition_masks(n)]


def oracle(fn, n: int) -> list[F]:
    """Exact Shapley values: average marginal contribution over all n! orderings."""
    phi = [F(0)] * n
    orders = list(itertools.permutations(range(n)))
    for order in orders:
        seen: set[int] = set()
        for i in order:
            before = F(fn(frozenset(seen)))
            seen.add(i)
            phi[i] += F(fn(frozenset(seen))) - before
    return [p / len(orders) for p in phi]


def got(fn, n: int) -> np.ndarray:
    return np.asarray(shapley_values(np.array([float(x) for x in table(fn, n)])), dtype=np.float64)


def close(a, b, tol=1e-5) -> tuple[bool, str]:
    a, b = np.asarray(a, dtype=np.float64), np.array([float(x) for x in b])
    err = float(np.abs(a - b).max())
    return err <= tol, f"max error {err:.1e}"


def textbook_games() -> None:
    print("Textbook games with answers known on paper")

    # Glove game: players 0, 1 hold a left glove, player 2 a right one; a pair is worth 1.
    # Player 2 is pivotal unless it comes first (4 of 6 orderings) -> 2/3; the others split the rest.
    glove = lambda s: int(2 in s and (0 in s or 1 in s))  # noqa: E731
    check("glove game (1/6, 1/6, 2/3)", *close(got(glove, 3), [F(1, 6), F(1, 6), F(2, 3)]))

    # Weighted voting [quota 3; weights 2, 1, 1]: (2/3, 1/6, 1/6).
    vote3 = lambda s: int(sum((2, 1, 1)[i] for i in s) >= 3)  # noqa: E731
    check("voting [3; 2,1,1] (2/3, 1/6, 1/6)", *close(got(vote3, 3), [F(2, 3), F(1, 6), F(1, 6)]))

    # Weighted voting [4; 3,1,1,1,1]: the big player is pivotal in positions 2, 3, 4 of 5 -> 3/5; the rest 1/10 each.
    vote5 = lambda s: int(sum((3, 1, 1, 1, 1)[i] for i in s) >= 4)  # noqa: E731
    check("voting [4; 3,1,1,1,1] (3/5, 1/10 x4)", *close(got(vote5, 5), [F(3, 5)] + [F(1, 10)] * 4))

    # Airport game, costs 1..5, v(S) = max cost in S: each runway segment is split among those who need it.
    # phi = (1/5, 1/5+1/4, 1/5+1/4+1/3, ... ) = (0.2, 0.45, 0.7833, 1.2833, 2.2833).
    airport = lambda s: max((i + 1 for i in s), default=0)  # noqa: E731
    expect = [sum(F(1, 5 - k) for k in range(i + 1)) for i in range(5)]
    check("airport game, costs 1..5", *close(got(airport, 5), expect))

    # Unanimity games: v(S) = 1 iff T is inside S. Members of T get 1/|T|, everyone else 0. All 31 T for n = 5.
    worst, ok = 0.0, True
    for mask in range(1, 32):
        t = frozenset(i for i in range(5) if mask >> i & 1)
        expect = [F(1, len(t)) if i in t else F(0) for i in range(5)]
        good, _ = close(g := got(lambda s, t=t: int(t <= s), 5), expect)
        worst, ok = max(worst, float(np.abs(g - np.array([float(x) for x in expect])).max())), ok and good
    check("all 31 unanimity games on 5 players", ok, f"max error {worst:.1e}")


def random_games() -> None:
    print("Random games against the exact brute force, and the axioms")
    rng = np.random.default_rng(7)
    worst = 0.0
    for n in range(1, 6):
        for _ in range(40):
            vals = {frozenset(np.flatnonzero(r).tolist()): F(int(rng.integers(-1000, 1000)), 8) for r in coalition_masks(n)}
            fn = vals.__getitem__
            worst = max(worst, float(np.abs(got(fn, n) - np.array([float(x) for x in oracle(fn, n)])).max()))
    check("200 random games, n = 1..5 (v(empty) != 0)", worst < 1e-3, f"max error {worst:.1e} on values up to 125")

    v, w = rng.normal(size=32), rng.normal(size=32)
    phi = lambda x: np.asarray(shapley_values(x), dtype=np.float64)  # noqa: E731
    check("efficiency: sum = v(all) - v(none)", abs(phi(v).sum() - (v[31] - v[0])) < 1e-5)
    check("linearity: phi(2v - 3w) = 2 phi(v) - 3 phi(w)", np.abs(phi(2 * v - 3 * w) - (2 * phi(v) - 3 * phi(w))).max() < 1e-5)
    check("a constant shift of every v(S) changes nothing", np.abs(phi(v + 1234.5) - phi(v)).max() < 1e-3)

    # Symmetry: make players 1 and 3 interchangeable by averaging v over the swap.
    masks = coalition_masks(5)
    swapped = masks.copy()
    swapped[:, [1, 3]] = masks[:, [3, 1]]
    index = (swapped * (1 << np.arange(5))).sum(1)
    sym = (v + v[index]) / 2
    check("symmetry: interchangeable players get the same value", abs(phi(sym)[1] - phi(sym)[3]) < 1e-5)

    # Null player: v ignores player 2.
    null = v[np.arange(32) & ~(1 << 2)]
    check("null player gets 0", abs(phi(null)[2]) < 1e-5)

    # Precision at the scale of real returns (thousands), where jax's float32 matters.
    big = rng.normal(-3000, 700, size=32)
    exact = oracle(lambda s: F(float(big[sum(1 << i for i in s)])), 5)
    err = float(np.abs(phi(big) - np.array([float(x) for x in exact])).max())
    check("float32 error at return scale (-3000 +/- 700)", err < 0.05, f"max error {err:.1e} points")

    loo = np.asarray(loo_values(v), dtype=np.float64)
    check("leave-one-out = v(all) - v(all minus i)", np.abs(loo - np.array([v[31] - v[31 ^ (1 << i)] for i in range(5)])).max() < 1e-5)


def rollout_checks(steps: int, keys: int) -> None:
    """Planted culprit on the real simulator: only one agent ever deviates from the reference."""
    import jax
    import jax.numpy as jnp

    from cc4lab import env as E
    from cc4lab import rollout as R
    from cc4lab.policies.trivial import always_sleep, random_valid

    print(f"Real rollouts ({steps} steps, {keys} keys, reference = Sleep)")
    env = E.make_env()
    _, state = E.reset(env, jax.random.PRNGKey(3))
    ks = jax.random.split(jax.random.PRNGKey(4), keys)
    masks = jnp.asarray(coalition_masks(E.N_BLUE))

    for culprit in (0, 3, 4):
        only = jnp.arange(E.N_BLUE) == culprit

        def actual(k, o, m, only=only):  # the culprit acts at random, everyone else already equals the reference
            return jnp.where(only, random_valid(k, o, m), always_sleep(k, o, m))

        v = np.asarray(R.make_coalition_returns_fn(env, actual, always_sleep, steps)(state, ks, masks), dtype=np.float64)
        phi = np.asarray(shapley_values(v), dtype=np.float64)
        others = np.delete(phi, culprit, axis=0)
        gap = v[31] - v[0]
        check(f"culprit {culprit}: the other four get exactly 0", float(np.abs(others).max()) == 0.0,
              f"max |phi| {np.abs(others).max():.1e}")
        check(f"culprit {culprit}: it gets the whole gap v(all) - v(none)", np.abs(phi[culprit] - gap).max() < 1e-2,
              f"gap per key {np.round(gap, 1).tolist()}")

    v = np.asarray(R.make_coalition_returns_fn(env, random_valid, always_sleep, steps)(state, ks, masks), dtype=np.float64)
    phi = np.asarray(shapley_values(v), dtype=np.float64)
    exact = np.array([[float(x) for x in oracle(lambda s, j=j: F(float(v[sum(1 << i for i in s), j])), 5)] for j in range(keys)]).T
    check("all five random: matches the exact brute force per key", np.abs(phi - exact).max() < 1e-2,
          f"max error {np.abs(phi - exact).max():.1e}")
    check("all five random: efficiency per key", np.abs(phi.sum(0) - (v[31] - v[0])).max() < 1e-2,
          f"max error {np.abs(phi.sum(0) - (v[31] - v[0])).max():.1e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", action="store_true", help="also run the planted-culprit checks on the simulator (compiles for a while)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--keys", type=int, default=3)
    args = p.parse_args()
    textbook_games()
    random_games()
    if args.rollouts:
        rollout_checks(args.steps, args.keys)
    print(f"\n{'ALL CHECKS PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
