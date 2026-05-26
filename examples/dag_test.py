import os

os.environ["JAX_NUM_CPU_DEVICES"] = "8"

import jax
import jax.numpy as jnp
import numpy as np

from stak import (
    Node,
    _reshard_dep_for_child,
    compile_hierarchy,
    device_group,
    hierarchy_model,
)


def image_source_model(x):
    # Local x shape: [B_local, 1, 32, 32]
    return x


def vector_source_model(x):
    # Local x shape: [B_local, 10]
    return x


def sample_scalar_source_model(x):
    # Local x shape: [B_local]
    return x


def global_scalar_source_model(x):
    # Local x shape: []
    return x


def tensor_source_model(x):
    # Local x shape: [B_local, 4, 6, 8]
    return x


def latent_from_vector_and_scalar(args):
    # args["vector"] local shape:        [B_local, 10]
    # args["sample_scalar"] local shape: [B_local]
    # args["global_scalar"] local shape: []
    vector = args["vector"]
    sample_scalar = args["sample_scalar"]
    global_scalar = args["global_scalar"]

    return vector + sample_scalar[:, None] + global_scalar


def image_conditioned_on_latent(args):
    # args["image"] local shape:  [B_local, 1, 32, 32]
    # args["latent"] local shape: [B_local, 10]
    image = args["image"]
    latent = args["latent"]

    conditioning = latent[:, 0, None, None, None]

    return image + conditioning


def final_model(args):
    # args["conditioned_image"] local shape: [B_local, 1, 32, 32]
    # args["tensor"] local shape:            [B_local, 4, 6, 8]
    conditioned_image = args["conditioned_image"]
    tensor = args["tensor"]

    image_summary = jnp.mean(conditioned_image, axis=(1, 2, 3))
    tensor_summary = jnp.mean(tensor, axis=(1, 2, 3))

    return image_summary + tensor_summary


def print_device_summary(devices):
    print("Visible JAX devices:")
    print(devices)
    print(f"Number of visible JAX devices: {len(devices)}")

    if len(devices) < 8:
        print(
            "\nWARNING: JAX sees fewer than 8 devices. "
            "If you expected 8 CPU devices, run this script before any other "
            "JAX code starts in the same process.\n"
        )


def print_hierarchy_summary(hierarchy, order):
    print("\nTopological order:")
    print(order)

    print("\nHierarchy mesh/spec summary:")
    for name in order:
        entry = hierarchy[name]
        print(
            f"{name:18s}",
            "deps =",
            entry.deps,
            "| devices =",
            len(entry.devices),
            "| mesh_shape =",
            entry.mesh_shape,
            "| out_specs =",
            entry.out_specs,
        )


def build_hierarchy(devices):
    return dict(
        image=Node(
            name="image",
            devices=device_group(devices, 0, 2),
            event_shape=(1, 32, 32),
            has_sample_axis=True,
            deps=(),
            fn=image_source_model,
        ),
        vector=Node(
            name="vector",
            devices=device_group(devices, 2, 4),
            event_shape=(10,),
            has_sample_axis=True,
            deps=(),
            fn=vector_source_model,
        ),
        sample_scalar=Node(
            name="sample_scalar",
            devices=device_group(devices, 4, 6),
            event_shape=(),
            has_sample_axis=True,
            deps=(),
            fn=sample_scalar_source_model,
        ),
        global_scalar=Node(
            name="global_scalar",
            devices=device_group(devices, 4, 6),
            event_shape=(),
            has_sample_axis=False,
            deps=(),
            fn=global_scalar_source_model,
        ),
        tensor=Node(
            name="tensor",
            devices=device_group(devices, 6, 8),
            event_shape=(4, 6, 8),
            has_sample_axis=True,
            deps=(),
            fn=tensor_source_model,
        ),
        latent=Node(
            name="latent",
            devices=device_group(devices, 6, 8),
            event_shape=(10,),
            output_event_shape=(10,),
            has_sample_axis=True,
            deps=("vector", "sample_scalar", "global_scalar"),
            fn=latent_from_vector_and_scalar,
        ),
        conditioned_image=Node(
            name="conditioned_image",
            devices=device_group(devices, 0, 2),
            event_shape=(1, 32, 32),
            output_event_shape=(1, 32, 32),
            has_sample_axis=True,
            deps=("image", "latent"),
            fn=image_conditioned_on_latent,
        ),
        final=Node(
            name="final",
            devices=device_group(devices, 2, 4),
            event_shape=(),
            output_event_shape=(),
            has_sample_axis=True,
            deps=("conditioned_image", "tensor"),
            fn=final_model,
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

    batch_size = 10
    inputs = dict(
        image=jnp.ones((batch_size, 1, 32, 32)),
        vector=jnp.ones((batch_size, 10)),
        sample_scalar=jnp.ones((batch_size,)),
        global_scalar=jnp.array(1.0),
        tensor=jnp.ones((batch_size, 4, 6, 8)),
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
