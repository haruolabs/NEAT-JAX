"""Backprop NEAT example on 2D classification tasks."""

import argparse
import json
from functools import partial
from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from PIL import Image

from src.draw import draw
from src.evaluator import SimpleEvaluator
from src.genome import Genome
from src.lineage import EvolutionLineage, collect_ancestry, trace_primary_lineage
from src.population import NEATConfig
from src.topology import build_topology_and_weights, topology2policy
from src.trainer import EvolutionResult, evolve

jax.config.update("jax_platforms", "cuda")

print(f"Using device: {jax.devices()[0]}")


def generate_circles_dataset(key: jax.Array, n_samples: int = 500, noise: float = 0.1):
    """Generate concentric circles dataset (inner circle = class 0, outer = class 1)."""
    key, k1, k2, k3, k4, k5 = jr.split(key, 6)

    # Inner circle
    n_inner = n_samples // 2
    r_inner = jr.uniform(k1, (n_inner,), minval=0.0, maxval=0.3)
    theta_inner = jr.uniform(k2, (n_inner,), minval=0.0, maxval=2 * jnp.pi)
    x_inner = r_inner * jnp.cos(theta_inner)
    y_inner = r_inner * jnp.sin(theta_inner)
    labels_inner = jnp.zeros(n_inner)

    # Outer circle
    n_outer = n_samples - n_inner
    r_outer = jr.uniform(k3, (n_outer,), minval=0.6, maxval=1.0)
    theta_outer = jr.uniform(k4, (n_outer,), minval=0.0, maxval=2 * jnp.pi)
    x_outer = r_outer * jnp.cos(theta_outer)
    y_outer = r_outer * jnp.sin(theta_outer)
    labels_outer = jnp.ones(n_outer)

    # Combine and add noise
    X = jnp.stack([
        jnp.concatenate([x_inner, x_outer]),
        jnp.concatenate([y_inner, y_outer]),
    ], axis=1)
    X = X + jr.normal(k5, X.shape) * noise
    y = jnp.concatenate([labels_inner, labels_outer])
    return X, y


def binary_cross_entropy_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy loss.

    Args:
        predictions: [N, 1] raw network outputs
        targets: [N] binary labels (0 or 1)
    """
    # Apply sigmoid to get probabilities
    probs = jax.nn.sigmoid(predictions.squeeze())
    # Clip to avoid log(0)
    probs = jnp.clip(probs, 1e-7, 1.0 - 1e-7)
    # Binary cross-entropy
    loss = -jnp.mean(targets * jnp.log(probs) + (1 - targets) * jnp.log(1 - probs))
    return loss


def predict_probabilities(genome: Genome, X: jnp.ndarray) -> jnp.ndarray:
    """Run the genome on inputs and return sigmoid probabilities."""
    topology, weights = build_topology_and_weights(genome)
    policy = topology2policy(topology)
    raw_outputs = policy(weights, X)
    return jax.nn.sigmoid(raw_outputs.squeeze())


def compute_accuracy(genome: Genome, data: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
    """Compute binary classification accuracy."""
    X, y = data
    predictions = predict_probabilities(genome, X)
    predicted_labels = (predictions > 0.5).astype(jnp.float32)
    accuracy = jnp.mean(predicted_labels == y).item()
    return float(accuracy)


def evaluate_genome(genome: Genome, key: jax.Array, test_data: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
    """Evaluate genome fitness on test data."""
    del key
    accuracy = compute_accuracy(genome, test_data)
    fitness = accuracy - 0.1 * jnp.log(1 + genome.num_parameters)
    return float(fitness)


def save_genome(genome: Genome, save_path: Path) -> None:
    """Persist a genome as JSON so it can be reproduced later."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(genome.to_dict(), f, indent=2)


def plot_decision_boundary(
    genome: Genome,
    data: Tuple[jnp.ndarray, jnp.ndarray],
    save_path: Path,
    grid_size: int = 300,
) -> None:
    """Plot the classifier probability field and validation points."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for decision-boundary plots") from exc

    X, y = data
    x_min, x_max = float(X[:, 0].min()) - 0.2, float(X[:, 0].max()) + 0.2
    y_min, y_max = float(X[:, 1].min()) - 0.2, float(X[:, 1].max()) + 0.2

    xx, yy = jnp.meshgrid(
        jnp.linspace(x_min, x_max, grid_size),
        jnp.linspace(y_min, y_max, grid_size),
    )
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    probs = predict_probabilities(genome, grid).reshape(xx.shape)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, levels=30, cmap="RdBu", alpha=0.75)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.5)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="white", linewidths=0.6)
    fig.colorbar(contour, ax=ax, label="P(class=1)")
    ax.set_title("Decision Boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(*scatter.legend_elements(), title="Class", loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_lineage_history(lineage: EvolutionLineage, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(lineage.to_dict(), f, indent=2)


def save_gif(image_paths: List[Path], save_path: Path, duration_ms: int = 350) -> None:
    if not image_paths:
        return
    frames = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
    finally:
        for frame in frames:
            frame.close()


def save_lineage_artifacts(
    result: EvolutionResult,
    best_genome_id: int,
    data: Tuple[jnp.ndarray, jnp.ndarray],
    output_dir: Path,
) -> None:
    if result.lineage is None:
        raise ValueError("No lineage information is available in the evolution result")

    lineage_dir = output_dir / "lineage"
    topology_dir = lineage_dir / "topology_snapshots"
    boundary_dir = lineage_dir / "decision_boundaries"
    summary_path = lineage_dir / "lineage_summary.json"

    topology_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)

    ancestry_records = collect_ancestry(result.lineage, best_genome_id)
    primary_records = trace_primary_lineage(result.lineage, best_genome_id)

    boundary_paths: List[Path] = []
    for record in primary_records:
        genome = Genome.from_dict(record.genome)
        topology, weights = build_topology_and_weights(genome)
        stem = f"gen_{record.generation:03d}_genome_{record.genome_id}"
        topology_path = topology_dir / f"{stem}.png"
        boundary_path = boundary_dir / f"{stem}.png"
        draw(topology, weights, save_path=str(topology_path))
        plot_decision_boundary(genome, data, boundary_path)
        boundary_paths.append(boundary_path)

    save_gif(boundary_paths, lineage_dir / "decision_boundary_lineage.gif")

    summary = {
        "final_best_genome_id": best_genome_id,
        "primary_lineage_genome_ids": [record.genome_id for record in primary_records],
        "ancestry_genome_ids": [record.genome_id for record in ancestry_records],
        "topology_snapshot_dir": str(topology_dir),
        "decision_boundary_dir": str(boundary_dir),
        "decision_boundary_gif": str(lineage_dir / "decision_boundary_lineage.gif"),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-best-genome",
        action="store_true",
        help="Save the best genome as JSON after evolution.",
    )
    parser.add_argument(
        "--plot-decision-boundary",
        action="store_true",
        help="Render and save a decision-boundary plot for the best genome.",
    )
    parser.add_argument(
        "--save-lineage-history",
        action="store_true",
        help="Save explicit genome ancestry and species history as JSON.",
    )
    parser.add_argument(
        "--visualize-lineage",
        action="store_true",
        help="Save topology snapshots, decision-boundary frames, and a GIF for the champion's primary lineage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/classification"),
        help="Directory used for optional outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Generate dataset
    key = jr.PRNGKey(42)
    key, train_key, test_key, val_key = jr.split(key, 4)
    X_train, y_train = generate_circles_dataset(train_key, n_samples=400, noise=0.05)
    X_test, y_test = generate_circles_dataset(test_key, n_samples=200, noise=0.05)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Configure NEAT with backprop
    config = NEATConfig(
        pop_size=100, # 50
        delta_threshold=5.0, # 8.0
        enable_backprop=True,
        backprop_steps=100, # 100
        backprop_lr=0.01,
        backprop_batch_size=128,
        # Mutate more
        p_mutate_add_connection = 0.2,
        p_mutate_add_node = 0.2,
        target_fitness=0.9,

    )
    GENERATIONS = 20 # 100

    # Create evaluator (evaluates on test set)
    test_data = (X_test, y_test)
    eval_fn = partial(evaluate_genome, test_data=test_data)
    evaluator = SimpleEvaluator(eval_fn)

    # Training data for backprop
    train_data = (X_train, y_train)

    # Run evolution
    print("Starting evolution with backprop...")
    result = evolve(
        n_inputs=2,
        n_outputs=1,
        evaluator=evaluator,
        key=key,
        config=config,
        generations=GENERATIONS,
        add_bias=True,
        verbose=True,
        loss_fn=binary_cross_entropy_loss,
        train_data=train_data,
    )

    print("Evolution Complete!")

    # Analyze best genome
    fitness_history = [h.best_fitness for h in result.history]
    best_idx = int(jnp.argmax(jnp.array(fitness_history)))
    best_genome = result.history[best_idx].best_genome
    best_fitness = fitness_history[best_idx]
    best_genome_id = result.history[best_idx].best_genome_id
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Best genome ID: {best_genome_id}")

    # Validate the best genome
    X_val, y_val = generate_circles_dataset(val_key, n_samples=200, noise=0.05)
    val_data = (X_val, y_val)
    val_accuracy = compute_accuracy(best_genome, val_data)
    val_fitness = evaluate_genome(best_genome, val_key, val_data)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation fitness: {val_fitness:.4f}")

    if args.save_best_genome:
        genome_path = args.output_dir / "best_genome.json"
        save_genome(best_genome, genome_path)
        print(f"Saved best genome to {genome_path}")

    if args.plot_decision_boundary:
        plot_path = args.output_dir / "decision_boundary.png"
        plot_decision_boundary(best_genome, val_data, plot_path)
        print(f"Saved decision-boundary plot to {plot_path}")

    if args.save_lineage_history or args.visualize_lineage:
        lineage_path = args.output_dir / "lineage_history.json"
        if result.lineage is None:
            raise ValueError("The evolution result did not include lineage history")
        save_lineage_history(result.lineage, lineage_path)
        print(f"Saved lineage history to {lineage_path}")

    if args.visualize_lineage:
        if best_genome_id is None:
            raise ValueError("The best genome does not have a lineage ID")
        save_lineage_artifacts(result, best_genome_id, val_data, args.output_dir)
        print(f"Saved lineage visualizations under {args.output_dir / 'lineage'}")
