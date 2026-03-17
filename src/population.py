from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp
import jax.random as jr

from src.evaluator import Evaluator

from .genome import Genome
from .innovation import InnovationTracker
from .lineage import EvolutionLineage, GenomeLineageRecord, SpeciesLineageRecord


@dataclass
class NEATConfig:
    pop_size: int = 100
    # speciation
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    delta_threshold: float = 3.0
    # reproduction
    elite_per_species: int = 1
    crossover_prob: float = 0.75
    # mutation
    p_mutate_weights: float = 0.9
    p_mutate_add_connection: float = 0.05
    p_mutate_add_node: float = 0.05
    p_mutate_toggle_connection: float = 0.01
    weight_sigma: float = 0.5
    weight_reset_prob: float = 0.1
    w_init_std: float = 1.0
    # backpropagation
    enable_backprop: bool = False
    backprop_steps: int = 100
    backprop_lr: float = 0.01
    backprop_batch_size: Optional[int] = None  # None = full batch
    target_fitness: Optional[float] = None


@dataclass
class Species:
    species_id: int
    representative: int
    members: List[int]


class Population:
    def __init__(
        self,
        genomes: List[Genome],
        tracker: InnovationTracker,
        key: jax.Array,
        config: NEATConfig,
        *,
        genome_ids: Optional[List[int]] = None,
        genome_parent_ids: Optional[List[List[int]]] = None,
        genome_origins: Optional[List[str]] = None,
        next_genome_id: Optional[int] = None,
        next_species_id: int = 0,
        lineage: Optional[EvolutionLineage] = None,
    ):
        self.genomes = genomes
        self.tracker = tracker
        self.key = key
        self.config = config
        self.fitness: List[float] = [0.0] * len(genomes)
        self.species: List[Species] = []
        self.generation: int = 0
        self.genome_ids = list(genome_ids) if genome_ids is not None else list(range(len(genomes)))
        self.genome_parent_ids = [list(parent_ids) for parent_ids in genome_parent_ids] if genome_parent_ids is not None else [[] for _ in genomes]
        self.genome_origins = list(genome_origins) if genome_origins is not None else ["initial"] * len(genomes)
        self.next_genome_id = next_genome_id if next_genome_id is not None else len(self.genome_ids)
        self.next_species_id = next_species_id
        self.lineage = lineage or EvolutionLineage()
        self._last_recorded_generation: Optional[int] = None

    @staticmethod
    def from_initial_feedforward(
        n_inputs: int,
        n_outputs: int,
        key: jax.Array,
        config: Optional[NEATConfig] = None,
        add_bias: bool = True,
    ) -> "Population":
        config = config or NEATConfig()
        tracker = InnovationTracker()
        genomes: List[Genome] = []
        
        # 1) Build a single template with global node/innov IDs
        k0, key = jr.split(key)
        template_genome = Genome.from_initial_feedforward(
            n_inputs, n_outputs, tracker=tracker, key=k0, add_bias=add_bias, w_init_std=1.0
        )
        
        # 2) Copy it pop_size times and reinit weights per copy
        for i in range(config.pop_size):
            key, k = jr.split(key)
            genome = template_genome.copy()
            genome.mutate_weights(k, p_reset=1.0, w_init_std=config.w_init_std)
            genomes.append(genome)

        genome_ids = list(range(config.pop_size))
        genome_parent_ids = [[] for _ in range(config.pop_size)]
        genome_origins = ["initial"] * config.pop_size
        return Population(
            genomes,
            tracker,
            key,
            config,
            genome_ids=genome_ids,
            genome_parent_ids=genome_parent_ids,
            genome_origins=genome_origins,
            next_genome_id=config.pop_size,
            next_species_id=0,
            lineage=EvolutionLineage(),
        )

    def _allocate_genome_id(self) -> int:
        genome_id = self.next_genome_id
        self.next_genome_id += 1
        return genome_id

    def evaluate(self, evaluator: Evaluator) -> None:
        """Modern evaluation method using strategy pattern."""
        self.fitness = evaluator.evaluate(self.genomes, self.key)

    def speciate(self) -> None:
        if not self.genomes:
            self.species = []
            return

        # Distance cache (i, j) -> distance
        dist_cache = {}

        def dist(i: int, j: int) -> float:
            a, b = (i, j) if i <= j else (j, i)
            if (a, b) not in dist_cache:
                d = self.genomes[a].compatibility_distance(self.genomes[b], self.config.c1, self.config.c2, self.config.c3)
                dist_cache[(a, b)] = d
            return dist_cache[(a, b)]

        unspeciated = set(range(len(self.genomes)))
        new_rep_indices: List[int] = []
        new_members: List[List[int]] = []
        new_species_ids: List[int] = []

        for species in self.species:
            if not unspeciated:
                break
            rep_idx = min(unspeciated, key=lambda gid: dist(gid, species.representative))
            new_rep_indices.append(rep_idx)
            new_members.append([rep_idx])
            new_species_ids.append(species.species_id)
            unspeciated.remove(rep_idx)
        
        # 2) Assign remaining genomes to the closest species under threshold; otherwise create a new species.
        while unspeciated:
            gid = unspeciated.pop()
            candidates = []
            for species_idx, rep_idx in enumerate(new_rep_indices):
                delta = dist(gid, rep_idx)
                if delta <= self.config.delta_threshold:
                    candidates.append((delta, species_idx))
            if candidates:
                _, species_idx = min(candidates, key=lambda item: item[0])
                new_members[species_idx].append(gid)
            else:
                new_rep_indices.append(gid)
                new_members.append([gid])
                new_species_ids.append(self.next_species_id)
                self.next_species_id += 1
        
        # 3) Rebuild species list with chosen representatives and members.
        self.species = [
            Species(species_id=species_id, representative=rep_idx, members=members)
            for species_id, rep_idx, members in zip(new_species_ids, new_rep_indices, new_members)
        ]

    def record_lineage_snapshot(self) -> None:
        if self._last_recorded_generation == self.generation:
            return

        species_by_genome = {}
        for species in self.species:
            for genome_idx in species.members:
                species_by_genome[genome_idx] = species.species_id

        for genome_idx, genome in enumerate(self.genomes):
            self.lineage.genome_records.append(
                GenomeLineageRecord(
                    generation=self.generation,
                    genome_id=self.genome_ids[genome_idx],
                    species_id=species_by_genome.get(genome_idx),
                    fitness=float(self.fitness[genome_idx]),
                    parent_genome_ids=list(self.genome_parent_ids[genome_idx]),
                    origin=self.genome_origins[genome_idx],
                    num_parameters=genome.num_parameters,
                    genome=genome.to_dict(),
                )
            )

        for species in self.species:
            best_idx = max(species.members, key=lambda gid: self.fitness[gid])
            avg_fitness = sum(self.fitness[gid] for gid in species.members) / len(species.members)
            self.lineage.species_records.append(
                SpeciesLineageRecord(
                    generation=self.generation,
                    species_id=species.species_id,
                    representative_genome_id=self.genome_ids[species.representative],
                    member_genome_ids=[self.genome_ids[gid] for gid in species.members],
                    best_genome_id=self.genome_ids[best_idx],
                    best_fitness=float(self.fitness[best_idx]),
                    avg_fitness=float(avg_fitness),
                )
            )

        self.lineage.final_generation = self.generation
        self._last_recorded_generation = self.generation

    def _adjust_fitness(self) -> List[float]:
        adjusted_fitness = [0.0] * len(self.genomes)
        assert self.fitness, Exception("Fitness values should be defined for all genomes (even if they are zero)")
        min_fitness = min(self.fitness)
        shift = (-min_fitness + 1e-8) if min_fitness < 0 else 0.0

        for species in self.species:
            assert len(species.members) > 0
            size = len(species.members)
            for gid in species.members:
                fitness = self.fitness[gid] + shift
                adjusted_fitness[gid] = fitness / size
        return adjusted_fitness

    def reproduce(self) -> None:
        """Evolve the population for one generation using NEAT reproduction.
        
        This method implements the core NEAT reproduction algorithm:
        1. Speciation: Group genomes into species based on compatibility
        2. Fitness sharing: Adjust fitness within species to promote diversity
        3. Selection: Allocate offspring to species based on adjusted fitness
        4. Reproduction: Create new genomes via crossover and mutation
        
        The reproduction process maintains diversity through speciation while
        promoting improvement through fitness-based selection and genetic operators.
        """
        # Prepare per-generation innovation sharing
        self.tracker.new_gen()
        
        # divide into species
        self.speciate()
        
        # adjust fitness
        adjusted_fitness = self._adjust_fitness()
        
        # species adjusted total (sum of adjusted fitness members)
        species_adj = [sum(adjusted_fitness[gid] for gid in s.members) for s in self.species]
        total_adj = sum(species_adj)

        key = self.key
        new_genomes: List[Genome] = []
        new_genome_ids: List[int] = []
        new_genome_parent_ids: List[List[int]] = []
        new_genome_origins: List[str] = []

        if total_adj <= 0 or not self.species:
            for _ in range(self.config.pop_size):
                key, k_sel, k_mw, k_ac_b, k_ac, k_an_b, k_an = jr.split(key, 7)
                parent_idx = int(jr.randint(k_sel, (), 0, len(self.genomes)))
                parent = self.genomes[parent_idx]
                child = parent.copy()
                child.mutate_weights(k_mw)
                if jr.bernoulli(k_ac_b, self.config.p_mutate_add_connection):
                    child.mutate_add_connection(k_ac, self.tracker)
                if jr.bernoulli(k_an_b, self.config.p_mutate_add_node):
                    child.mutate_add_node(k_an, self.tracker)
                new_genomes.append(child)
                new_genome_ids.append(self._allocate_genome_id())
                new_genome_parent_ids.append([self.genome_ids[parent_idx]])
                new_genome_origins.append("mutation")

            self.genomes = new_genomes
            self.genome_ids = new_genome_ids
            self.genome_parent_ids = new_genome_parent_ids
            self.genome_origins = new_genome_origins
            self.fitness = [0.0] * len(new_genomes)
            self.key = key
            self.generation += 1
            self._last_recorded_generation = None
            return

        for species in self.species:
            if not species.members:
                continue
            sorted_members = sorted(species.members, key=lambda gid: self.fitness[gid], reverse=True)
            for elite_idx in sorted_members[: min(self.config.elite_per_species, len(sorted_members))]:
                new_genomes.append(self.genomes[elite_idx].copy())
                new_genome_ids.append(self._allocate_genome_id())
                new_genome_parent_ids.append([self.genome_ids[elite_idx]])
                new_genome_origins.append("elite")

        num_offspring = self.config.pop_size - len(new_genomes)
        offspring_per_species: List[int] = []
        if total_adj > 0:
            for species_total in species_adj:
                offspring_per_species.append(round(num_offspring * (species_total / total_adj)))
        else:
            offspring_per_species = [num_offspring // len(self.species)] * len(self.species)

        current_total = sum(offspring_per_species)
        while current_total < num_offspring:
            best_species_idx = max(range(len(species_adj)), key=lambda idx: species_adj[idx])
            offspring_per_species[best_species_idx] += 1
            current_total += 1

        while current_total > num_offspring:
            worst_species_idx = min([idx for idx, count in enumerate(offspring_per_species) if count > 0], key=lambda idx: species_adj[idx])
            offspring_per_species[worst_species_idx] -= 1
            current_total -= 1

        for species, n_offspring in zip(self.species, offspring_per_species):
            if n_offspring == 0 or not species.members:
                continue

            member_fitness = [adjusted_fitness[gid] for gid in species.members]
            total_member_fitness = sum(member_fitness)
            if total_member_fitness > 0:
                probs = [fitness / total_member_fitness for fitness in member_fitness]
            else:
                probs = [1.0 / len(species.members)] * len(species.members)

            for _ in range(n_offspring):
                key, k_sel, k_cross, k_mut = jr.split(key, 4)

                if len(species.members) == 1 or jr.bernoulli(k_sel, 1.0 - self.config.crossover_prob):
                    parent_idx = int(jr.choice(k_sel, len(species.members), p=jnp.array(probs)))
                    population_idx = species.members[parent_idx]
                    child = self.genomes[population_idx].copy()
                    parent_ids = [self.genome_ids[population_idx]]
                    origin = "mutation"
                else:
                    indices = jr.choice(k_sel, len(species.members), shape=(2,), p=jnp.array(probs), replace=False)
                    p1_idx, p2_idx = int(indices[0]), int(indices[1])
                    population_idx_1 = species.members[p1_idx]
                    population_idx_2 = species.members[p2_idx]
                    parent_1 = self.genomes[population_idx_1]
                    parent_2 = self.genomes[population_idx_2]
                    fitness_1 = self.fitness[population_idx_1]
                    fitness_2 = self.fitness[population_idx_2]
                    child = parent_1.crossover(parent_2, fitness_1, fitness_2, k_cross)
                    if fitness_2 > fitness_1:
                        parent_ids = [self.genome_ids[population_idx_2], self.genome_ids[population_idx_1]]
                    else:
                        parent_ids = [self.genome_ids[population_idx_1], self.genome_ids[population_idx_2]]
                    origin = "crossover"

                k_mw, k_ac_b, k_ac, k_an_b, k_an, k_tc_b, k_tc = jr.split(k_mut, 7)
                
                # Weight mutation: most common, fine-tunes existing connections
                if jr.bernoulli(k_mw, self.config.p_mutate_weights):
                    child.mutate_weights(k_mw, self.config.weight_sigma, self.config.weight_reset_prob)
                
                # Structural mutations: add complexity to the network
                if jr.bernoulli(k_ac_b, self.config.p_mutate_add_connection):
                    child.mutate_add_connection(k_ac, self.tracker)
                if jr.bernoulli(k_an_b, self.config.p_mutate_add_node):
                    child.mutate_add_node(k_an, self.tracker)
                
                # Connection state mutation: modify network topology
                if jr.bernoulli(k_tc_b, self.config.p_mutate_toggle_connection):
                    child.mutate_toggle_connection(k_tc)

                new_genomes.append(child)
                new_genome_ids.append(self._allocate_genome_id())
                new_genome_parent_ids.append(parent_ids)
                new_genome_origins.append(origin)

        self.genomes = new_genomes
        self.genome_ids = new_genome_ids
        self.genome_parent_ids = new_genome_parent_ids
        self.genome_origins = new_genome_origins
        self.fitness = [0.0] * len(new_genomes)
        self.key = key
        self.generation += 1
        self._last_recorded_generation = None
