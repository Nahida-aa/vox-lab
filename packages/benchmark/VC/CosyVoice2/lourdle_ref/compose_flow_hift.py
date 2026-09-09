import argparse

import onnx
from onnx.compose import merge_models

def strip_prefixes(model, input_prefix="flow_", output_prefix="hift_"):
    rename_map = {}

    def collect_renames(value_infos, prefix):
        for value_info in value_infos:
            if value_info.name.startswith(prefix):
                old = value_info.name
                new = old[len(prefix):]
                rename_map[old] = new
                value_info.name = new

    def apply_renames_to_graph(graph):
        for value_info in graph.input:
            if value_info.name in rename_map:
                value_info.name = rename_map[value_info.name]
        for value_info in graph.output:
            if value_info.name in rename_map:
                value_info.name = rename_map[value_info.name]
        for value_info in graph.value_info:
            if value_info.name in rename_map:
                value_info.name = rename_map[value_info.name]
        for initializer in graph.initializer:
            if initializer.name in rename_map:
                initializer.name = rename_map[initializer.name]
        for initializer in graph.sparse_initializer:
            if initializer.name in rename_map:
                initializer.name = rename_map[initializer.name]
        for node in graph.node:
            node.input[:] = [rename_map.get(name, name) for name in node.input]
            node.output[:] = [rename_map.get(name, name) for name in node.output]
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    apply_renames_to_graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        apply_renames_to_graph(subgraph)

    collect_renames(model.graph.input, input_prefix)
    collect_renames(model.graph.output, output_prefix)
    apply_renames_to_graph(model.graph)

    onnx.checker.check_model(model)

def main():
    parser = argparse.ArgumentParser(description="Combine Flow and HiFT ONNX modules")
    parser.add_argument('--flow_path', type=str, required=True, help='Path to the Flow ONNX module')
    parser.add_argument('--hift_path', type=str, required=True, help='Path to the HiFT ONNX module')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the combined ONNX module')
    args = parser.parse_args()

    flow = onnx.load(args.flow_path)
    hift = onnx.load(args.hift_path)

    assert flow.graph.output
    assert len(hift.graph.input) == 1

    flow_output_name = flow.graph.output[0].name
    hift_input_name = hift.graph.input[0].name

    model = merge_models(
        flow,
        hift,
        io_map=[(flow_output_name, hift_input_name)],
        prefix1="flow_",
        prefix2="hift_",
    )

    strip_prefixes(model, input_prefix="flow_", output_prefix="hift_")

    onnx.save(model, args.output_path)


if __name__ == "__main__":
    main()
