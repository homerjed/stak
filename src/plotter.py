from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from stak import RandomVariableEntry, topological_order


def hierarchy_edges(
    hierarchy: Mapping[str, RandomVariableEntry],
    order: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Return dependency edges as (parent, child) pairs."""
    if order is None:
        order = topological_order(hierarchy)

    edges = []

    for child_name in order:
        for parent_name in hierarchy[child_name].deps:
            edges.append((parent_name, child_name))

    return edges


def hierarchy_levels(
    hierarchy: Mapping[str, RandomVariableEntry],
    order: Sequence[str] | None = None,
) -> dict[str, int]:
    """Assign each node to a dependency depth for left-to-right drawings."""
    if order is None:
        order = topological_order(hierarchy)

    levels = {}

    for name in order:
        deps = hierarchy[name].deps
        levels[name] = 0 if len(deps) == 0 else 1 + max(levels[dep] for dep in deps)

    return levels


def hierarchy_to_mermaid(
    hierarchy: Mapping[str, RandomVariableEntry],
    order: Sequence[str] | None = None,
    *,
    direction: str = "LR",
    include_specs: bool = True,
) -> str:
    """Render the hierarchy as a Mermaid flowchart string."""
    if order is None:
        order = topological_order(hierarchy)

    lines = [f"flowchart {direction}"]

    for name in order:
        entry = hierarchy[name]
        label = name

        if include_specs:
            label = (
                f"{name}<br/>"
                f"mesh={entry.mesh_shape}<br/>"
                f"out={entry.out_specs}"
            )

        lines.append(f'    {name}["{label}"]')

    for parent_name, child_name in hierarchy_edges(hierarchy, order):
        lines.append(f"    {parent_name} --> {child_name}")

    return "\n".join(lines)


def hierarchy_to_dot(
    hierarchy: Mapping[str, RandomVariableEntry],
    order: Sequence[str] | None = None,
    *,
    include_specs: bool = True,
) -> str:
    """Render the hierarchy as a Graphviz DOT graph string."""
    if order is None:
        order = topological_order(hierarchy)

    lines = [
        "digraph hierarchy {",
        "    rankdir=LR;",
        '    node [shape=box, style="rounded"];',
    ]

    levels = hierarchy_levels(hierarchy, order)

    for name in order:
        entry = hierarchy[name]
        label = name

        if include_specs:
            label = (
                f"{name}\\n"
                f"mesh={entry.mesh_shape}\\n"
                f"out={entry.out_specs}"
            )

        lines.append(f'    "{name}" [label="{label}"];')

    for level in sorted(set(levels.values())):
        names_at_level = [name for name in order if levels[name] == level]
        same_rank = "; ".join(f'"{name}"' for name in names_at_level)
        lines.append(f"    {{ rank=same; {same_rank}; }}")

    for parent_name, child_name in hierarchy_edges(hierarchy, order):
        lines.append(f'    "{parent_name}" -> "{child_name}";')

    lines.append("}")

    return "\n".join(lines)


def write_hierarchy_diagram(
    hierarchy: Mapping[str, RandomVariableEntry],
    path: str | Path,
    order: Sequence[str] | None = None,
    *,
    format: str | None = None,
    include_specs: bool = True,
) -> Path:
    """Write a Mermaid (.mmd/.md) or Graphviz DOT (.dot/.gv) diagram."""
    path = Path(path)
    diagram_format = format

    if diagram_format is None:
        suffix = path.suffix.lower()
        if suffix in {".dot", ".gv"}:
            diagram_format = "dot"
        else:
            diagram_format = "mermaid"

    if diagram_format == "dot":
        text = hierarchy_to_dot(hierarchy, order, include_specs=include_specs)
    elif diagram_format == "mermaid":
        mermaid = hierarchy_to_mermaid(
            hierarchy,
            order,
            include_specs=include_specs,
        )
        text = mermaid if path.suffix == ".mmd" else f"```mermaid\n{mermaid}\n```\n"
    else:
        raise ValueError(f"Unknown diagram format {diagram_format!r}.")

    path.write_text(text)

    return path
