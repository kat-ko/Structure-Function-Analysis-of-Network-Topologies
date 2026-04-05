"""
Minimal NSGA-II for small binary genomes (Clune-style second objective on routing structure).

Both objectives are **minimized**: ``f1 = val MSE``, ``f2 = -routing_separation`` (higher separation
→ lower f2).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class Individual:
    genome: List[int]
    f1: float = float("inf")
    f2: float = float("inf")
    rank: int = 0
    crowding: float = 0.0


def _dominates(a: Individual, b: Individual) -> bool:
    return (a.f1 <= b.f1 and a.f2 <= b.f2) and (a.f1 < b.f1 or a.f2 < b.f2)


def fast_non_dominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    for p in pop:
        p.dom_s = []  # type: ignore[attr-defined]
        p.dom_count = 0  # type: ignore[attr-defined]
        for q in pop:
            if _dominates(p, q):
                p.dom_s.append(q)  # type: ignore[attr-defined]
            elif _dominates(q, p):
                p.dom_count += 1  # type: ignore[attr-defined]

    fronts: List[List[Individual]] = [[]]
    for p in pop:
        if p.dom_count == 0:  # type: ignore[attr-defined]
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: List[Individual] = []
        for p in fronts[i]:
            for q in p.dom_s:  # type: ignore[attr-defined]
                q.dom_count -= 1  # type: ignore[attr-defined]
                if q.dom_count == 0:  # type: ignore[attr-defined]
                    q.rank = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(front: List[Individual]) -> None:
    n = len(front)
    if n == 0:
        return
    for p in front:
        p.crowding = 0.0
    for key in ("f1", "f2"):
        front.sort(key=lambda x: getattr(x, key))
        front[0].crowding = float("inf")
        front[-1].crowding = float("inf")
        fmin = getattr(front[0], key)
        fmax = getattr(front[-1], key)
        span = fmax - fmin if fmax != fmin else 1.0
        for j in range(1, n - 1):
            if front[j].crowding == float("inf"):
                continue
            prevv = getattr(front[j - 1], key)
            nextv = getattr(front[j + 1], key)
            front[j].crowding += (nextv - prevv) / span


def environmental_selection(combined: List[Individual], n_keep: int) -> List[Individual]:
    fronts = fast_non_dominated_sort(combined)
    new_pop: List[Individual] = []
    for fr in fronts:
        crowding_distance(fr)
        if len(new_pop) + len(fr) <= n_keep:
            new_pop.extend(fr)
        else:
            fr.sort(key=lambda x: -x.crowding)
            need = n_keep - len(new_pop)
            new_pop.extend(fr[:need])
            break
    return new_pop


def tournament_select(pop: List[Individual], k: int = 2) -> Individual:
    cand = random.sample(pop, k=min(k, len(pop)))
    cand.sort(key=lambda x: (x.rank, -x.crowding))
    return cand[0]


def crossover(a: List[int], b: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    if len(a) != len(b):
        raise ValueError("Genome length mismatch")
    if len(a) <= 1:
        return a[:], b[:]
    pt = rng.randint(1, len(a) - 1)
    c1 = a[:pt] + b[pt:]
    c2 = b[:pt] + a[pt:]
    return c1, c2


def mutate(g: List[int], p_flip: float, rng: random.Random) -> List[int]:
    return [1 - x if rng.random() < p_flip else x for x in g]


def make_offspring(parents: List[Individual], pop_size: int, rng: random.Random) -> List[Individual]:
    L = len(parents[0].genome)
    kids: List[Individual] = []
    while len(kids) < pop_size:
        p1 = tournament_select(parents)
        p2 = tournament_select(parents)
        c1, c2 = crossover(p1.genome, p2.genome, rng)
        c1 = mutate(c1, 1.0 / L, rng)
        c2 = mutate(c2, 1.0 / L, rng)
        kids.append(Individual(c1))
        if len(kids) < pop_size:
            kids.append(Individual(c2))
    return kids


def run_nsga2(
    evaluate: Callable[[List[int]], Tuple[float, float]],
    *,
    genome_length: int = 12,
    population_size: int = 32,
    generations: int = 20,
    seed: int = 0,
) -> List[Individual]:
    rng = random.Random(seed)
    pop: List[Individual] = [
        Individual([rng.randint(0, 1) for _ in range(genome_length)]) for _ in range(population_size)
    ]
    for ind in pop:
        ind.f1, ind.f2 = evaluate(ind.genome)

    for _ in range(generations - 1):
        fronts = fast_non_dominated_sort(pop)
        for fr in fronts:
            crowding_distance(fr)
        offspring = make_offspring(pop, population_size, rng)
        for ind in offspring:
            ind.f1, ind.f2 = evaluate(ind.genome)
        pop = environmental_selection(pop + offspring, population_size)

    fronts = fast_non_dominated_sort(pop)
    for fr in fronts:
        crowding_distance(fr)
    return pop
