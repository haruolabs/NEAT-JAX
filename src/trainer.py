from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from .evaluator import Evaluator
from .genome import Genome
from .lineage import EvolutionLineage
from .population import NEATConfig, Population


@dataclass
class EvolutionMetrics:
    generation: int
    best_fitness: float
    avg_fitness: float
    num_species: int
    mean_parameters: float
    mean_num_nodes: float
    best_genome: Genome
    best_genome_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "num_species": self.num_species,
            "mean_parameters": self.mean_parameters,
            "mean_num_nodes": self.mean_num_nodes,
            "best_genome": self.best_genome.to_dict(),
            "best_genome_id": self.best_genome_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionMetrics":
        """Create from dictionary."""
        return cls(
            generation=data["generation"],
            best_fitness=data["best_fitness"],
            avg_fitness=data["avg_fitness"],
            num_species=data["num_species"],
            mean_parameters=data["mean_parameters"],
            mean_num_nodes=data.get("mean_num_nodes", 0.0),
            best_genome=Genome.from_dict(data["best_genome"]),
            best_genome_id=data.get("best_genome_id"),
        )


@dataclass
class EvolutionResult:
    population: Population
    history: List[EvolutionMetrics]
    lineage: Optional[EvolutionLineage] = None


def evolve(
    n_inputs: int,
    n_outputs: int,
    evaluator: Evaluator,
    *,
    key: jax.Array,
    config: Optional[NEATConfig] = None,
    generations: int = 100,
    add_bias: bool = True,
    verbose: bool = True,
    loss_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    train_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
) -> EvolutionResult:
    """Run the NEAT evolutionary loop for a fixed number of generations."""
    cfg = config or NEATConfig()

    key, pop_key = jr.split(key)
    population = Population.from_initial_feedforward(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        key=pop_key,
        config=cfg,
        add_bias=add_bias,
    )

    history: List[EvolutionMetrics] = []

    for gen in range(generations):
        if cfg.enable_backprop:
            if loss_fn is None or train_data is None:
                raise ValueError("loss_fn and train_data must be provided when enable_backprop=True")

            from .backprop import optimize_weights

            for genome in population.genomes:
                optimize_weights(
                    genome,
                    loss_fn,
                    train_data,
                    cfg.backprop_steps,
                    cfg.backprop_lr,
                    cfg.backprop_batch_size,
                )

        population.evaluate(evaluator)
        population.speciate()

        assert population.fitness, Exception("Fitness values should be defined for all genomes (even if they are zero)")
        best_idx = int(jnp.argmax(jnp.array(population.fitness)))
        best_fitness = max(population.fitness)
        avg_fitness = sum(population.fitness) / len(population.fitness)
        num_species = len(population.species)
        mean_parameters = sum(genome.num_parameters for genome in population.genomes) / len(population.genomes)
        mean_num_nodes = sum(len(genome.nodes) for genome in population.genomes) / len(population.genomes)

        metrics = EvolutionMetrics(
            generation=gen,
            best_fitness=float(best_fitness),
            avg_fitness=float(avg_fitness),
            num_species=int(num_species),
            mean_parameters=float(mean_parameters),
            mean_num_nodes=float(mean_num_nodes),
            best_genome=population.genomes[best_idx],
            best_genome_id=population.genome_ids[best_idx],
        )
        history.append(metrics)
        population.record_lineage_snapshot()

        if verbose:
            print(
                f"Gen {gen:03d} | Best Fitness: {best_fitness:6.4f} | Avg Fitness: {avg_fitness:6.4f} | "
                f"Species: {num_species} | Mean Parameters: {mean_parameters:6.4f} | Avg Num of Nodes: {mean_num_nodes:6.4f}"
            )

        if cfg.target_fitness is not None and best_fitness >= cfg.target_fitness:
            break

        if gen < generations - 1:
            population.reproduce()

    return EvolutionResult(population=population, history=history, lineage=population.lineage)
