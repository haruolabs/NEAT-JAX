from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenomeLineageRecord:
    generation: int
    genome_id: int
    species_id: Optional[int]
    fitness: float
    parent_genome_ids: List[int]
    origin: str
    num_parameters: int
    genome: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "genome_id": self.genome_id,
            "species_id": self.species_id,
            "fitness": self.fitness,
            "parent_genome_ids": list(self.parent_genome_ids),
            "origin": self.origin,
            "num_parameters": self.num_parameters,
            "genome": self.genome,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenomeLineageRecord":
        return cls(
            generation=int(data["generation"]),
            genome_id=int(data["genome_id"]),
            species_id=data.get("species_id"),
            fitness=float(data["fitness"]),
            parent_genome_ids=[int(gid) for gid in data.get("parent_genome_ids", [])],
            origin=str(data["origin"]),
            num_parameters=int(data["num_parameters"]),
            genome=dict(data["genome"]),
        )


@dataclass
class SpeciesLineageRecord:
    generation: int
    species_id: int
    representative_genome_id: int
    member_genome_ids: List[int]
    best_genome_id: int
    best_fitness: float
    avg_fitness: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "species_id": self.species_id,
            "representative_genome_id": self.representative_genome_id,
            "member_genome_ids": list(self.member_genome_ids),
            "best_genome_id": self.best_genome_id,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeciesLineageRecord":
        return cls(
            generation=int(data["generation"]),
            species_id=int(data["species_id"]),
            representative_genome_id=int(data["representative_genome_id"]),
            member_genome_ids=[int(gid) for gid in data.get("member_genome_ids", [])],
            best_genome_id=int(data["best_genome_id"]),
            best_fitness=float(data["best_fitness"]),
            avg_fitness=float(data["avg_fitness"]),
        )


@dataclass
class EvolutionLineage:
    genome_records: List[GenomeLineageRecord] = field(default_factory=list)
    species_records: List[SpeciesLineageRecord] = field(default_factory=list)
    final_generation: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_generation": self.final_generation,
            "genome_records": [record.to_dict() for record in self.genome_records],
            "species_records": [record.to_dict() for record in self.species_records],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionLineage":
        return cls(
            genome_records=[GenomeLineageRecord.from_dict(record) for record in data.get("genome_records", [])],
            species_records=[SpeciesLineageRecord.from_dict(record) for record in data.get("species_records", [])],
            final_generation=data.get("final_generation"),
        )


def build_genome_record_map(lineage: EvolutionLineage) -> Dict[int, GenomeLineageRecord]:
    return {record.genome_id: record for record in lineage.genome_records}


def collect_ancestry(lineage: EvolutionLineage, genome_id: int) -> List[GenomeLineageRecord]:
    record_map = build_genome_record_map(lineage)
    visited: set[int] = set()
    ordered: List[GenomeLineageRecord] = []

    def visit(current_id: int) -> None:
        if current_id in visited or current_id not in record_map:
            return
        visited.add(current_id)
        record = record_map[current_id]
        for parent_id in record.parent_genome_ids:
            visit(parent_id)
        ordered.append(record)

    visit(genome_id)
    ordered.sort(key=lambda record: (record.generation, record.genome_id))
    return ordered


def trace_primary_lineage(lineage: EvolutionLineage, genome_id: int) -> List[GenomeLineageRecord]:
    record_map = build_genome_record_map(lineage)
    ordered: List[GenomeLineageRecord] = []
    visited: set[int] = set()
    current_id: Optional[int] = genome_id

    while current_id is not None and current_id not in visited and current_id in record_map:
        visited.add(current_id)
        record = record_map[current_id]
        ordered.append(record)
        current_id = record.parent_genome_ids[0] if record.parent_genome_ids else None

    ordered.reverse()
    return ordered
