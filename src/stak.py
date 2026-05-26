# ---------------------------------------------------------------------
# DAG hierarchy solution
#
# Key idea:
# - Source nodes have deps=() and consume external inputs[name].
# - Derived nodes have deps=("a", "b", ...) and consume a dict:
#       {"a": value_a, "b": value_b, ...}
# - The hierarchy is evaluated in topological order.
# - Parent values are explicitly resharded onto the child node's mesh.
# ---------------------------------------------------------------------

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


Array = jax.Array
ModelFn = Callable[[Any], Array]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _resource_size(resource: Any, mesh_axis_sizes: dict[str, int]) -> int:
    if resource is None:
        return 1

    if isinstance(resource, str):
        return mesh_axis_sizes[resource]

    if isinstance(resource, tuple):
        size = 1
        for axis in resource:
            size *= mesh_axis_sizes[axis]
        return size

    raise TypeError(f"Invalid sharding resource: {resource!r}")


def _spec_to_tuple(spec: P) -> tuple[Any, ...]:
    return tuple(spec)


def device_group(
    devices: Sequence[Any],
    start: int,
    stop: int,
    *,
    allow_reuse_if_insufficient: bool = True,
) -> tuple[Any, ...]:
    """Return the device slice assigned to one DAG node.

    For example, device_group(devices, 2, 4) returns devices[2:4]. If that
    slice is empty, the default fallback reuses the first available devices so
    the demo can still run on smaller machines. Set allow_reuse_if_insufficient
    to False when overlapping device groups should be treated as an error.
    """
    devices = tuple(devices)

    if len(devices) == 0:
        raise RuntimeError("JAX reports zero devices.")

    group = tuple(devices[start:stop])

    if len(group) > 0:
        return group

    if not allow_reuse_if_insufficient:
        raise ValueError(
            f"Requested devices[{start}:{stop}], but JAX only sees "
            f"{len(devices)} device(s): {devices}."
        )

    requested_size = max(1, stop - start)
    fallback_size = min(requested_size, len(devices))

    return tuple(devices[:fallback_size])


def _identity(x):
    return x


def _default_float_dtype():
    return jnp.asarray(0.0).dtype


def _cast_tree(x, dtype):
    if dtype is None:
        return x

    def _cast(y):
        y = (
            y.astype(dtype)
            if hasattr(y, "astype")
            and hasattr(y, "dtype")
            and jnp.issubdtype(y.dtype, jnp.floating)
            else y
        )
        return y

    return jax.tree_util.tree_map(_cast, x)


def _is_tracing(x):
    leaves = jax.tree_util.tree_leaves(x)
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _shard_value(x, sharding):
    if _is_tracing(x):
        return lax.with_sharding_constraint(x, sharding)

    return jax.device_put(x, sharding)


def _axis_resources_for_value(
    *,
    has_sample_axis: bool,
    event_shape: tuple[int, ...],
    target_mesh_axes: tuple[str, ...],
) -> tuple[Any, ...]:
    """Build a default PartitionSpec payload for one random-variable value.

    Batched values have shape [B, *event_shape], so their leading sample axis is
    sharded over mesh axis "n" and event dimensions are replicated. Global
    values have no sample axis, so all dimensions are replicated by default.
    """
    if has_sample_axis:
        if "n" not in target_mesh_axes:
            raise ValueError(
                "Cannot feed a batched dependency into a target mesh "
                'without an "n" axis.'
            )

        return ("n",) + (None,) * len(event_shape)

    return (None,) * len(event_shape)


def _validate_array_against_specs(
    *,
    name: str,
    x: Array,
    has_sample_axis: bool,
    event_shape: tuple[int, ...],
    axis_resources: tuple[Any, ...],
    mesh_axis_sizes: dict[str, int],
):
    """Check that an array shape is compatible with its sharding spec.

    The value must have the expected rank, the expected event/global shape, one
    axis-resource entry per array dimension, and each sharded dimension must be
    divisible by the number of mesh devices used for that dimension.
    """
    expected_rank = len(event_shape) + int(has_sample_axis)

    # Check that the array rank matches [B, *event_shape] or event_shape.
    if x.ndim != expected_rank:
        expected = f"[B, *{event_shape}]" if has_sample_axis else f"{event_shape}"
        raise ValueError(
            f"{name}: expected rank {expected_rank}, shape {expected}, "
            f"got {x.shape}."
        )

    # Check that the non-sample dimensions match the declared value shape.
    if has_sample_axis:
        if x.shape[1:] != event_shape:
            raise ValueError(
                f"{name}: expected event shape {event_shape}, got {x.shape[1:]}."
            )
    else:
        if x.shape != event_shape:
            raise ValueError(
                f"{name}: expected global/unbatched shape {event_shape}, "
                f"got {x.shape}."
            )

    # Check that the sharding spec has one resource entry per array dimension.
    if len(axis_resources) != x.ndim:
        raise ValueError(
            f"{name}: axis_resources has length {len(axis_resources)}, "
            f"but array has rank {x.ndim}."
        )

    # Check that every sharded dimension can split evenly across its mesh axes.
    for dim, resource in zip(x.shape, axis_resources):
        shard_factor = _resource_size(resource, mesh_axis_sizes)

        if dim % shard_factor != 0:
            raise ValueError(
                f"{name}: dimension size {dim} is not divisible by shard factor "
                f"{shard_factor} from resource {resource!r}."
            )


# ---------------------------------------------------------------------
# DAG node
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    name: str
    devices: tuple[Any, ...]

    # Shape of one random variable value, excluding optional leading sample axis.
    #
    # If has_sample_axis=True:
    #     full value shape is [B, *event_shape]
    #
    # If has_sample_axis=False:
    #     full value shape is [*event_shape]
    event_shape: tuple[int, ...] = ()
    has_sample_axis: bool = True

    # Parent nodes in the DAG.
    #
    # deps=() means this is a source node and consumes inputs[name].
    # deps=("a", "b") means this node's model receives:
    #     {"a": values["a"], "b": values["b"]}
    deps: tuple[str, ...] = ()

    # Function run inside shard_map.
    #
    # For source nodes:
    #     fn(x) -> output
    #
    # For derived nodes:
    #     fn({"dep1": x1, "dep2": x2, ...}) -> output
    fn: ModelFn | None = None

    # None means "use JAX's configured default float dtype" when the node is
    # constructed, so jax_enable_x64=True gives float64 by default.
    dtype: Any = None

    mesh_axes: tuple[str, ...] = ("n",)
    mesh_shape: tuple[int, ...] | None = None

    # For source node input.
    in_axis_resources: tuple[Any, ...] | None = None

    # Output event shape. If None, equals event_shape.
    output_event_shape: tuple[int, ...] | None = None
    out_axis_resources: tuple[Any, ...] | None = None

    # Optional custom dependency sharding specs.
    #
    # Example:
    #   dep_axis_resources={
    #       "image": ("n", None, None, None),
    #       "global_scalar": (),
    #   }
    dep_axis_resources: Mapping[str, tuple[Any, ...]] | None = None

    apply: ModelFn | None = field(default=None, repr=False, compare=False)
    arg_specs: Any = field(default=None, repr=False, compare=False)

    mesh: Mesh = field(init=False, repr=False)
    mesh_axis_sizes: dict[str, int] = field(init=False, repr=False)

    in_specs: P = field(init=False)
    out_specs: P = field(init=False)

    in_sharding: NamedSharding = field(init=False, repr=False)
    out_sharding: NamedSharding = field(init=False, repr=False)

    def __post_init__(self):
        devices = tuple(self.devices)

        if len(devices) == 0:
            raise ValueError(f"{self.name}: got an empty device list.")

        object.__setattr__(self, "devices", devices)
        object.__setattr__(
            self,
            "dtype",
            _default_float_dtype() if self.dtype is None else self.dtype,
        )

        # Use a one-dimensional mesh by default, with all devices on axis "n".
        if self.mesh_shape is None:
            mesh_shape = (len(devices),) + (1,) * (len(self.mesh_axes) - 1)
        else:
            mesh_shape = self.mesh_shape

        # Check that each mesh dimension has at least one device.
        if any(dim <= 0 for dim in mesh_shape):
            raise ValueError(
                f"{self.name}: mesh_shape must have only positive dimensions, "
                f"got {mesh_shape}."
            )

        # Check that the mesh shape and named mesh axes describe the same rank.
        if len(mesh_shape) != len(self.mesh_axes):
            raise ValueError(
                f"{self.name}: mesh_shape {mesh_shape} must have the same rank as "
                f"mesh_axes {self.mesh_axes}."
            )

        # Check that the mesh shape accounts for exactly the provided devices.
        if int(np.prod(mesh_shape)) != len(devices):
            raise ValueError(
                f"{self.name}: mesh_shape {mesh_shape} has size {np.prod(mesh_shape)}, "
                f"but got {len(devices)} device(s)."
            )

        # Batched random variables need a mesh axis named "n" for samples.
        if self.has_sample_axis and "n" not in self.mesh_axes:
            raise ValueError(
                f'{self.name}: mesh_axes must contain sample axis "n" '
                f"when has_sample_axis=True."
            )

        device_array = np.asarray(devices, dtype=object).reshape(mesh_shape)
        mesh = Mesh(device_array, self.mesh_axes)
        mesh_axis_sizes = dict(zip(self.mesh_axes, mesh_shape))

        input_rank = len(self.event_shape) + int(self.has_sample_axis)

        # Use the standard sample-axis sharding unless the input spec is custom.
        if self.in_axis_resources is None:
            in_axis_resources = _axis_resources_for_value(
                has_sample_axis=self.has_sample_axis,
                event_shape=self.event_shape,
                target_mesh_axes=self.mesh_axes,
            )
        else:
            in_axis_resources = self.in_axis_resources

        # Check that the input sharding spec has one entry per input dimension.
        if len(in_axis_resources) != input_rank:
            raise ValueError(
                f"{self.name}: in_axis_resources must have length {input_rank}, "
                f"got {in_axis_resources}."
            )

        output_event_shape = (
            self.event_shape
            if self.output_event_shape is None
            else self.output_event_shape
        )

        output_rank = len(output_event_shape) + int(self.has_sample_axis)

        # Reuse the input sharding for same-rank outputs unless overridden.
        if self.out_axis_resources is None:
            if output_rank == input_rank:
                out_axis_resources = in_axis_resources
            else:
                out_axis_resources = _axis_resources_for_value(
                    has_sample_axis=self.has_sample_axis,
                    event_shape=output_event_shape,
                    target_mesh_axes=self.mesh_axes,
                )
        else:
            out_axis_resources = self.out_axis_resources

        # Check that the output sharding spec has one entry per output dimension.
        if len(out_axis_resources) != output_rank:
            raise ValueError(
                f"{self.name}: out_axis_resources must have length {output_rank}, "
                f"got {out_axis_resources}."
            )

        event_offset = int(self.has_sample_axis)

        for dim_size, resource in zip(
            self.event_shape,
            in_axis_resources[event_offset:],
        ):
            shard_factor = _resource_size(resource, mesh_axis_sizes)

            # Check that input event dimensions split evenly over sharded axes.
            if dim_size % shard_factor != 0:
                raise ValueError(
                    f"{self.name}: event dimension size {dim_size} is not divisible "
                    f"by shard factor {shard_factor} from resource {resource!r}."
                )

        for dim_size, resource in zip(
            output_event_shape,
            out_axis_resources[event_offset:],
        ):
            shard_factor = _resource_size(resource, mesh_axis_sizes)

            # Check that output event dimensions split evenly over sharded axes.
            if dim_size % shard_factor != 0:
                raise ValueError(
                    f"{self.name}: output event dimension size {dim_size} is not "
                    f"divisible by shard factor {shard_factor} from resource "
                    f"{resource!r}."
                )

        in_specs = P(*in_axis_resources)
        out_specs = P(*out_axis_resources)

        object.__setattr__(self, "mesh_shape", mesh_shape)
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "mesh_axis_sizes", mesh_axis_sizes)

        object.__setattr__(self, "output_event_shape", output_event_shape)
        object.__setattr__(self, "in_axis_resources", in_axis_resources)
        object.__setattr__(self, "out_axis_resources", out_axis_resources)

        object.__setattr__(self, "in_specs", in_specs)
        object.__setattr__(self, "out_specs", out_specs)

        object.__setattr__(self, "in_sharding", NamedSharding(mesh, in_specs))
        object.__setattr__(self, "out_sharding", NamedSharding(mesh, out_specs))

    def validate_source_input(self, x: Array):
        _validate_array_against_specs(
            name=self.name,
            x=x,
            has_sample_axis=self.has_sample_axis,
            event_shape=self.event_shape,
            axis_resources=self.in_axis_resources,
            mesh_axis_sizes=self.mesh_axis_sizes,
        )

    def default_dep_specs(self, dep_entry: "Node") -> P:
        axis_resources = _axis_resources_for_value(
            has_sample_axis=dep_entry.has_sample_axis,
            event_shape=dep_entry.output_event_shape,
            target_mesh_axes=self.mesh_axes,
        )

        return P(*axis_resources)

    def compile(
        self,
        hierarchy: Mapping[str, "Node"],
    ) -> "Node":
        fn = self.fn if self.fn is not None else _identity

        # Source nodes receive one external array, so shard_map uses this node's
        # input PartitionSpec directly.
        if len(self.deps) == 0:
            arg_specs = self.in_specs
        else:
            arg_specs = {}

            for dep_name in self.deps:
                if dep_name not in hierarchy:
                    raise KeyError(
                        f"{self.name}: dependency {dep_name!r} is not in hierarchy."
                    )

                dep_entry = hierarchy[dep_name]

                # Use a custom dependency sharding spec if this child declares
                # one for the parent; otherwise derive the default from the
                # parent value shape.
                if (
                    self.dep_axis_resources is not None
                    and dep_name in self.dep_axis_resources
                ):
                    axis_resources = self.dep_axis_resources[dep_name]
                    spec = P(*axis_resources)
                else:
                    spec = self.default_dep_specs(dep_entry)

                arg_specs[dep_name] = spec

        @jax.shard_map(
            mesh=self.mesh,
            in_specs=(arg_specs,),
            out_specs=self.out_specs,
        )
        def sharded_fn(arg):
            arg = _cast_tree(arg, self.dtype)
            return fn(arg)

        return replace(self, apply=sharded_fn, arg_specs=arg_specs)


# ---------------------------------------------------------------------
# Topological sorting and DAG evaluation
# ---------------------------------------------------------------------

def topological_order(hierarchy: Mapping[str, Node]) -> list[str]:
    order = []
    state = {}

    def visit(name: str):
        if name not in hierarchy:
            raise KeyError(f"Unknown node {name!r}.")

        status = state.get(name, "unvisited")

        if status == "visiting":
            raise ValueError(
                f"Cycle detected at node {name!r}. "
                f"This evaluator only supports DAGs."
            )

        if status == "visited":
            return

        state[name] = "visiting"

        # Check dependency exists   
        for dep in hierarchy[name].deps:
            if dep not in hierarchy:
                raise KeyError(
                    f"{name}: dependency {dep!r} is not present in hierarchy."
                )

            visit(dep)

        state[name] = "visited"
        order.append(name)

    for name in hierarchy:
        visit(name)

    return order


def compile_hierarchy(
    hierarchy: Mapping[str, Node],
) -> tuple[dict[str, Node], list[str]]:
    # Compile parents before children so dependency metadata is available when a
    # child builds its input specs.
    order = topological_order(hierarchy)

    compiled = {}

    for name in order:
        # Let this node see already-compiled parents while preserving metadata
        # for nodes that have not been compiled yet.
        metadata_view = {**hierarchy, **compiled}
        compiled[name] = hierarchy[name].compile(metadata_view)

    return compiled, order


def _reshard_dep_for_child(
    *,
    dep_name: str,
    dep_value: Array,
    dep_entry: Node,
    child_entry: Node,
    dep_spec: P,
) -> Array:
    dep_axis_resources = _spec_to_tuple(dep_spec)

    _validate_array_against_specs(
        name=f"{child_entry.name} dependency {dep_name}",
        x=dep_value,
        has_sample_axis=dep_entry.has_sample_axis,
        event_shape=dep_entry.output_event_shape,
        axis_resources=dep_axis_resources,
        mesh_axis_sizes=child_entry.mesh_axis_sizes,
    )

    sharding = NamedSharding(child_entry.mesh, dep_spec)

    return _shard_value(dep_value, sharding)


def hierarchy_model(
    inputs: Mapping[str, Array],
    hierarchy: Mapping[str, Node],
    order: Sequence[str] | None = None,
    *,
    return_all: bool = True,
) -> dict[str, Array]:
    if order is None:
        order = topological_order(hierarchy)

    values = {}

    for name in order:
        entry = hierarchy[name]

        # Check there is an input -> output model for this entry
        if entry.apply is None:
            raise ValueError(
                f"Hierarchy entry {name!r} is not compiled. "
                f"Call compile_hierarchy(hierarchy) first."
            )

        # Source nodes have no parents, so their value must come from inputs.
        if len(entry.deps) == 0:
            if name not in inputs:
                raise KeyError(f"Missing external input for source node {name!r}.")

            x = inputs[name]
            entry.validate_source_input(x)

            arg = _shard_value(x, entry.in_sharding)

        else:
            arg = {}

            for dep_name in entry.deps:
                dep_entry = hierarchy[dep_name]
                dep_value = values[dep_name]
                dep_spec = entry.arg_specs[dep_name]

                arg[dep_name] = _reshard_dep_for_child(
                    dep_name=dep_name,
                    dep_value=dep_value,
                    dep_entry=dep_entry,
                    child_entry=entry,
                    dep_spec=dep_spec,
                )

        values[name] = entry.apply(arg)

    if return_all:
        return values

    used_as_dep = set()

    for entry in hierarchy.values():
        used_as_dep.update(entry.deps)

    terminal_names = [name for name in order if name not in used_as_dep]

    return {name: values[name] for name in terminal_names}
