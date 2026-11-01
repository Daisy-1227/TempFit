import os
import json
import math
import argparse
import xml.etree.cElementTree as ET
from typing import Optional

import numpy as np
from svgpathtools import parse_path, svgstr2paths, wsvg, Line, Path

from utils.parallel_mapper import parallel_map
from utils.svg_util import (
    scan_dir,
    get_namespace,
    add_ns,
    del_ns,
    primitive2str,
    get_t_values,
)
from data.floorplancad.dataclass_define import ProcessArgs, SVGData


def clip_line_to_bbox(
    line_args: list[float], bbox: list[float]
) -> tuple[list[float], bool]:
    """
    Clip a line segment to a bounding box if it extends beyond the box.

    Args:
        line_args: [x1, y1, x2, y2] coordinates of the line
        bbox: [x_min, y_min, x_max, y_max] coordinates of the bounding box

    Returns:
        tuple containing:
          - [x1', y1', x2', y2'] coordinates of the clipped line
          - Boolean indicating if the line was clipped (True) or fully outside (False)
    """
    # Extract line coordinates
    x1, y1, x2, y2 = line_args
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Define line parameters
    dx = x2 - x1
    dy = y2 - y1

    # Initialize parameters
    t_min = 0.0
    t_max = 1.0

    # Check if line is parallel to an axis
    if dx == 0 and dy == 0:  # Line is a point
        # Check if point is inside the box
        if x_min <= x1 <= x_max and y_min <= y1 <= y_max:
            return line_args, False  # Point is inside, no clipping needed
        else:
            return [], True  # Point is outside, line is fully clipped

    # Check intersection with x_min boundary
    if dx != 0:
        t = (x_min - x1) / dx
        if dx > 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)
    elif x1 < x_min or x1 > x_max:
        return [], True  # Line is parallel to y-axis and outside x-bounds

    # Check intersection with x_max boundary
    if dx != 0:
        t = (x_max - x1) / dx
        if dx < 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)

    # Check intersection with y_min boundary
    if dy != 0:
        t = (y_min - y1) / dy
        if dy > 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)
    elif y1 < y_min or y1 > y_max:
        return [], True  # Line is parallel to x-axis and outside y-bounds

    # Check intersection with y_max boundary
    if dy != 0:
        t = (y_max - y1) / dy
        if dy < 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)

    # Check if line intersects with the box
    if t_min > t_max:
        return [], True  # Line doesn't intersect the box

    # Compute clipped line coordinates
    x1_new = x1 + t_min * dx
    y1_new = y1 + t_min * dy
    x2_new = x1 + t_max * dx
    y2_new = y1 + t_max * dy

    # Check if line was clipped
    was_clipped = t_min > 0 or t_max < 1

    return [x1_new, y1_new, x2_new, y2_new], was_clipped


def exceed_max_length(line_args_list: list[list[float]], max_length: float) -> bool:
    # line_args: [x1, y1, x2, y2]
    for line_args in line_args_list:
        length = np.linalg.norm(np.array(line_args[0:2]) - np.array(line_args[2:4]))
        if length - max_length > 1e-6:
            return True
    return False


def sample_primitive(
    prim_path: Path,
    is_origin_line: bool,
    line_t_values: list[float],
    curve_t_values: list[float],
    max_length: float,
    bbox: list[float],
    connect_lines: bool,
) -> list[list[float]]:
    t_values = line_t_values if is_origin_line else curve_t_values
    try_cnt = 0
    while True:
        try_cnt += 1
        line_args_list = []
        for i in range(len(t_values) - 1):
            line_args_list.append(
                [
                    prim_path.point(t_values[i]).real,
                    prim_path.point(t_values[i]).imag,
                    prim_path.point(t_values[i + 1]).real,
                    prim_path.point(t_values[i + 1]).imag,
                ]
            )
        if max_length > 0 and exceed_max_length(line_args_list, max_length):
            t_values = get_t_values(
                math.ceil(prim_path.length() / max_length) + try_cnt
            )
        else:
            break
    # clip line segments to bounding box
    cliped_line_args_list = []
    for line_args in line_args_list:
        cliped_line_args, _ = clip_line_to_bbox(line_args, bbox)
        if cliped_line_args:
            cliped_line_args_list.append(cliped_line_args)
    line_args_list = cliped_line_args_list
    if connect_lines:
        return line_args_list
    else:
        # convert line args to point args
        if len(line_args_list) == 0:
            return []
        point_args_list = []
        for line_args in line_args_list:
            point_args_list.append([line_args[0], line_args[1]])
        point_args_list.append(line_args_list[-1][2:])
        return point_args_list


def parse_primitive(
    primitive: ET.Element,
    line_t_values: list[float],
    curve_t_values: list[float],
    max_length: float,
    bbox: list[float],
    connect_lines: bool,
) -> tuple[list[list[float]], float, list[int], float]:
    # get primitive width
    prim_width = float(primitive.attrib.get("stroke-width", 0.1))
    # get primitive color
    prim_color = primitive.attrib.get("stroke", "rgb(0,0,0)")
    prim_color = list(map(int, prim_color.strip("rgb(").strip(")").split(",")))
    # parse primitive to svg object
    path: Optional[Path] = None
    is_origin_line = (
        False  # Flag to check if the primitive is originally a line segment
    )
    if del_ns(primitive.tag, get_namespace(primitive)) == "path":
        # Parse path string to svg object
        path_str = primitive.attrib.get("d", "")
        if path_str == "":
            raise ValueError("Path string is empty")
        else:
            path = parse_path(path_str)
        if path[0].__class__.__name__ == "Line":
            is_origin_line = True
    else:
        # Transform all svg shapes to svg object
        paths, _ = svgstr2paths(primitive2str(primitive))  # type: ignore
        path = paths[0]
    # get primitive length
    prim_length = path.length()
    if type(prim_length) != float:
        prim_length = 0.0
        raise ValueError("Primitive length is not a float")
    # Sample primitives
    sampled_coords = sample_primitive(
        path,
        is_origin_line,
        line_t_values,
        curve_t_values,
        max_length,
        bbox,
        connect_lines,
    )
    return sampled_coords, prim_width, prim_color, prim_length


def parse_svg(
    input_file_path: str,
    line_t_values: list[float],
    curve_t_values: list[float],
    connect_lines: bool,
    dynamic_sampling: bool,
    dynamic_sampling_ratio: float,
) -> SVGData:
    tree = ET.parse(input_file_path)
    root = tree.getroot()

    namespace = get_namespace(root)

    # Initialize SVG data class
    svg_data = SVGData(
        viewBox=[
            float(x) for x in root.attrib["viewBox"].split(" ")
        ],  # [x_min, y_min, width, height]
        coords=[],  # (n_points, 2) point coordinates or (n_lines, 4) line coordinates
        colors=[],  # (n_points, 3) or (n_lines, 3) color
        widths=[],  # (n_points,) or (n_lines,) width
        primitive_ids=[],  # (n_points,) or (n_lines,) primitive id, indicates which primitive the point or line belongs to
        layer_ids=[],  # (n_points,) or (n_lines,) layer id(start from 0), indicates which layer the point or line belongs to
        semantic_ids=[],  # (n_prims,) semantic id (start from 1 in FloorPlanCAD dataset)
        instance_ids=[],  # (n_prims,) instance id (start from 1 in FloorPlanCAD dataset)
        primitive_lengths=[],  # (n_prims,) primitive length
    )

    width, height = svg_data.viewBox[2], svg_data.viewBox[3]
    if dynamic_sampling:
        max_length = min(width, height) * dynamic_sampling_ratio
    else:
        max_length = -1
    x_min, y_min = svg_data.viewBox[0], svg_data.viewBox[1]
    x_max, y_max = x_min + width, y_min + height
    bbox = [x_min, y_min, x_max, y_max]

    layer_id = 0
    primitive_id = 0
    for group in root.iter(add_ns("g", namespace)):
        # Skip empty layer
        if len(group) == 0:
            continue
        for primitive in group:
            # ------------- sample primitives ------------ #
            sampled_coords, prim_width, prim_color, prim_length = parse_primitive(
                primitive,
                line_t_values,
                curve_t_values,
                max_length,
                bbox,
                connect_lines,
            )
            if len(sampled_coords) == 0:
                continue
            # ------------------ get ids ----------------- #
            # In FloorPlanCAD dataset, semanticId starts from 1 to 35, and instanceId starts from 1,
            # instanceId=-1 means uncountable semantic.
            # we define semanticId=36 for background and instanceId=-1 for background/uncountable
            semantic_id = int(primitive.attrib.get("semanticId", 36))
            if semantic_id == 36:  # background semantic
                instance_id = -1
            else:  # valid semantic
                instance_id = int(primitive.attrib.get("instanceId", -1))
            semantic_id -= (
                1  # shift id from [1, 36] to [0, 35] for better compatibility
            )
            # ---------------- append data --------------- #
            for coord in sampled_coords:
                svg_data.coords.append(coord)
                svg_data.colors.append(prim_color)
                svg_data.widths.append(prim_width)
                svg_data.primitive_ids.append(primitive_id)
                svg_data.layer_ids.append(layer_id)
            svg_data.semantic_ids.append(semantic_id)
            svg_data.instance_ids.append(instance_id)
            svg_data.primitive_lengths.append(prim_length)
            # -------------------------------------------- #
            primitive_id += 1
        layer_id += 1

    return svg_data


def save_svg(svg_data: SVGData, output_file_path: str) -> None:
    svg_attributes = {
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
        "viewBox": " ".join(map(str, svg_data.viewBox)),
    }
    paths = []
    colors = []
    stroke_widths = []
    nodes = []
    node_colors = []
    node_radii = []
    if len(svg_data.coords[0]) == 2:  # point coords
        for point_coord, prim_width, prim_color in zip(
            svg_data.coords, svg_data.widths, svg_data.colors
        ):
            stroke_color = tuple(prim_color)
            nodes.append(complex(point_coord[0], point_coord[1]))
            node_colors.append(stroke_color)
            node_radii.append(prim_width * 2)
    else:  # line coords
        for line_coord, prim_width, prim_color in zip(
            svg_data.coords, svg_data.widths, svg_data.colors
        ):
            stroke_color = tuple(prim_color)
            paths.append(
                Line(
                    complex(line_coord[0], line_coord[1]),
                    complex(line_coord[2], line_coord[3]),
                )
            )
            colors.append(stroke_color)
            stroke_widths.append(prim_width)
            nodes.append(complex(line_coord[0], line_coord[1]))
            nodes.append(complex(line_coord[2], line_coord[3]))
            node_colors.append(stroke_color)
            node_colors.append(stroke_color)
            node_radii.append(prim_width * 2)
            node_radii.append(prim_width * 2)
    wsvg(
        paths=paths,
        colors=colors,
        stroke_widths=stroke_widths,
        nodes=nodes,
        node_colors=node_colors,
        node_radii=node_radii,
        svg_attributes=svg_attributes,
        filename=output_file_path,
    )


def save_json(svg_data: SVGData, output_file_path: str) -> None:
    with open(output_file_path, "w") as f:
        f.write(json.dumps(svg_data.__dict__))


def process_svg(process_args: ProcessArgs) -> None:
    # Prepare input and output file paths
    input_file_path = os.path.join(process_args.input_dir, process_args.file_path)
    output_file_path = os.path.join(process_args.output_dir, process_args.file_path)
    # Parse svg data
    svg_data = parse_svg(
        input_file_path=input_file_path,
        line_t_values=process_args.line_t_values,
        curve_t_values=process_args.curve_t_values,
        connect_lines=process_args.connect_lines,
        dynamic_sampling=process_args.dynamic_sampling,
        dynamic_sampling_ratio=process_args.dynamic_sampling_ratio,
    )
    # Create output directory if not exists
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Save svg or json file
    if process_args.save_type == "svg":
        save_svg(svg_data, output_file_path)
    elif process_args.save_type == "json":
        output_file_path = output_file_path.replace(".svg", ".json")
        save_json(svg_data, output_file_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--save_type",
        type=str,
        default="json",
        choices=["json", "svg"],
        help="Determine whether to save svg or json file,"
        "`json`: save json file for training and evaluation,"
        "`svg`: save svg file for visualization",
    )
    parser.add_argument(
        "--connect_lines",
        action="store_true",
        default=False,
        help="Connect lines when saving svg or json file",
    )
    parser.add_argument(
        "--sample_lines",
        type=int,
        default=2,
        help="The number of sampling points sampled from a `line` primitive",
    )
    parser.add_argument(
        "--sample_curves",
        type=int,
        default=9,
        help="The number of sampling points sampled from a `curve` primitive",
    )
    parser.add_argument(
        "--dynamic_sampling", action="store_true", help="Enable dynamic sampling"
    )
    parser.add_argument(
        "--dynamic_sampling_ratio",
        type=float,
        default=0.01,
        help="Dynamic sampling ratio, controls the maximum `sampling point interval` when --dynamic_sampling is enabled. "
        "The process first uses the specified number of sampling points from --sample_lines and --sample_curves. "
        "If any `interval` exceeds `min(width, height) * dynamic_sampling_ratio`, "
        "the primitive will be resampled with additional sampling points until all intervals are shorter than this threshold. "
        "This ensures appropriate sampling density for large or complex primitives.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=64, help="Maximum number of workers"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Test mode: Randomly choose a subset of files, the subset ratio is specified by --test_ratio",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test ratio, only used when --test is True",
    )
    parser.add_argument(
        "--use_progress_bar",
        action="store_true",
        default=False,
        help="Use progress bar to monitor the progress.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Get all svg file paths
    svg_file_paths = scan_dir(args.input_dir, "svg")

    # Test mode: Randomly choose a subset of files
    if args.test:
        # Group by directory
        svg_file_paths_dict = {}
        for file_path in svg_file_paths:
            dir_name = os.path.dirname(file_path)
            if dir_name not in svg_file_paths_dict:
                svg_file_paths_dict[dir_name] = []
            svg_file_paths_dict[dir_name].append(file_path)
        # Randomly choose a subset of files
        for dir_name in svg_file_paths_dict:
            svg_file_paths_dict[dir_name] = np.random.choice(
                svg_file_paths_dict[dir_name],
                int(len(svg_file_paths_dict[dir_name]) * args.test_ratio),
                replace=False,
            )
        # merge grouped files
        svg_file_paths = []
        for dir_name in svg_file_paths_dict:
            svg_file_paths.extend(svg_file_paths_dict[dir_name])

    # Create output directory if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Prepare job arguments
    line_t_values = get_t_values(args.sample_lines)
    curve_t_values = get_t_values(args.sample_curves)
    job_args_list = [
        ProcessArgs(
            file_path=file_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            save_type=args.save_type,
            connect_lines=args.connect_lines,
            line_t_values=line_t_values,
            curve_t_values=curve_t_values,
            dynamic_sampling=args.dynamic_sampling,
            dynamic_sampling_ratio=args.dynamic_sampling_ratio,
        )
        for file_path in svg_file_paths
    ]

    # Process svg files in parallel
    parallel_map(
        process_svg,
        job_args_list,
        max_workers=args.max_workers,
        use_progress_bar=args.use_progress_bar,
    )


if __name__ == "__main__":
    main()
