import os
import sys
from pathlib import Path

os.environ["JAX_NUM_CPU_DEVICES"] = "8"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from plotter import write_hierarchy_diagram
from stak import (
    RandomVariableEntry,
    _reshard_dep_for_child,
    compile_hierarchy,
    device_group,
    hierarchy_model,
)


def image_source_model(x):
    # Local x shape: [B_local, C_local, H, W]
    return x


def channel_bias_source_model(x):
    # Local x shape: [C_local]
    return x


def channel_features_model(args):
    # args["image"] local shape: [B_local, C_local, H, W]
    image = args["image"]

    return jnp.mean(image, axis=(2, 3))


def channel_logits_model(args):
    # args["features"] local shape:     [B_local, C_local]
    # args["channel_bias"] local shape: [C_local]
    features = args["features"]
    channel_bias = args["channel_bias"]

    return features + channel_bias[None, :]


def final_score_model(args):
    # args["logits"] local shape: [B_local, C]
    logits = args["logits"]

    return jnp.mean(logits, axis=1)


def print_device_summary(devices):
    print("Visible JAX devices:")
    print(devices)
    print(f"Number of visible JAX devices: {len(devices)}")

    if len(devices) < 4:
        print(
            "\nWARNING: this example needs at least 4 visible devices for its "
            '("n", "c") mesh. With CPUs, run before other JAX code starts.\n'
        )


def print_hierarchy_summary(hierarchy, order):
    print("\nTopological order:")
    print(order)

    print("\nHierarchy mesh/spec summary:")
    for name in order:
        entry = hierarchy[name]
        print(
            f"{name:14s}",
            "deps =",
            entry.deps,
            "| mesh_axes =",
            entry.mesh_axes,
            "| mesh_shape =",
            entry.mesh_shape,
            "| in_specs =",
            entry.in_specs,
            "| out_specs =",
            entry.out_specs,
        )


def build_hierarchy(devices):
    image_devices = device_group(devices, 0, 4)
    channel_devices = device_group(devices, 0, 4)
    sample_devices = device_group(devices, 0, 2)

    return dict(
        image=RandomVariableEntry(
            name="image",
            devices=image_devices,
            event_shape=(4, 32, 32),
            has_sample_axis=True,
            deps=(),
            fn=image_source_model,
            mesh_axes=("n", "c"),
            mesh_shape=(2, 2),
            in_axis_resources=("n", "c", None, None),
            out_axis_resources=("n", "c", None, None),
        ),
        channel_bias=RandomVariableEntry(
            name="channel_bias",
            devices=channel_devices,
            event_shape=(4,),
            has_sample_axis=False,
            deps=(),
            fn=channel_bias_source_model,
            mesh_axes=("c",),
            mesh_shape=(4,),
            in_axis_resources=("c",),
            out_axis_resources=("c",),
        ),
        features=RandomVariableEntry(
            name="features",
            devices=image_devices,
            event_shape=(4, 32, 32),
            output_event_shape=(4,),
            has_sample_axis=True,
            deps=("image",),
            fn=channel_features_model,
            mesh_axes=("n", "c"),
            mesh_shape=(2, 2),
            in_axis_resources=("n", "c", None, None),
            out_axis_resources=("n", "c"),
            dep_axis_resources={
                "image": ("n", "c", None, None),
            },
        ),
        logits=RandomVariableEntry(
            name="logits",
            devices=image_devices,
            event_shape=(4,),
            output_event_shape=(4,),
            has_sample_axis=True,
            deps=("features", "channel_bias"),
            fn=channel_logits_model,
            mesh_axes=("n", "c"),
            mesh_shape=(2, 2),
            in_axis_resources=("n", "c"),
            out_axis_resources=("n", "c"),
            dep_axis_resources={
                "features": ("n", "c"),
                "channel_bias": ("c",),
            },
        ),
        final=RandomVariableEntry(
            name="final",
            devices=sample_devices,
            event_shape=(4,),
            output_event_shape=(),
            has_sample_axis=True,
            deps=("logits",),
            fn=final_score_model,
            mesh_axes=("n",),
            mesh_shape=(2,),
            in_axis_resources=("n", None),
            out_axis_resources=("n",),
            dep_axis_resources={
                "logits": ("n", None),
            },
        ),
    )


def hierarchy_model_with_jitted_node_calls(inputs, hierarchy, order):
    values = {}

    for name in order:
        entry = hierarchy[name]
        jitted_apply = jax.jit(entry.apply)

        if len(entry.deps) == 0:
            arg = jax.device_put(inputs[name], entry.in_sharding)
        else:
            arg = {}

            # Reshard each parent value into the input sharding expected by
            # this child node.
            for dep_name in entry.deps:
                arg[dep_name] = _reshard_dep_for_child(
                    dep_name=dep_name,
                    dep_value=values[dep_name],
                    dep_entry=hierarchy[dep_name],
                    child_entry=entry,
                    dep_spec=entry.arg_specs[dep_name],
                )

        values[name] = jitted_apply(arg)

    used_as_dep = set()

    for entry in hierarchy.values():
        used_as_dep.update(entry.deps)

    terminal_names = [name for name in order if name not in used_as_dep]

    return {name: values[name] for name in terminal_names}


def main():
    devices = tuple(jax.devices())
    print_device_summary(devices)

    hierarchy, order = compile_hierarchy(build_hierarchy(devices))
    print_hierarchy_summary(hierarchy, order)

    diagram_path = write_hierarchy_diagram(
        hierarchy,
        Path(__file__).with_suffix(".md"),
        order,
    )
    print(f"\nWrote hierarchy diagram: {diagram_path}")

    batch_size = 8
    channel_count = 4
    height = 32
    width = 32

    inputs = dict(
        image=jnp.ones((batch_size, channel_count, height, width)),
        channel_bias=jnp.arange(channel_count, dtype=jnp.float32),
    )

    outputs = hierarchy_model(inputs, hierarchy, order, return_all=True)

    print("\nOutput shapes:")
    print(jax.tree_util.tree_map(lambda x: x.shape, outputs))

    terminal_outputs = hierarchy_model(inputs, hierarchy, order, return_all=False)

    print("\nTerminal output only:")
    print(jax.tree_util.tree_map(lambda x: x.shape, terminal_outputs))

    jitted_terminal_outputs = hierarchy_model_with_jitted_node_calls(
        inputs,
        hierarchy,
        order,
    )

    jax.tree_util.tree_map(
        lambda eager, jitted: np.testing.assert_allclose(eager, jitted),
        terminal_outputs,
        jitted_terminal_outputs,
    )

    print("\nTerminal output with jitted node calls:")
    print(jax.tree_util.tree_map(lambda x: x.shape, jitted_terminal_outputs))

    print("\nOutput shardings:")
    print(jax.tree_util.tree_map(lambda x: x.sharding, outputs))


if __name__ == "__main__":
    main()
