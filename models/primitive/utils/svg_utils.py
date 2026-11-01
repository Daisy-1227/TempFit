import xml.etree.ElementTree as ET
import math, re
import numpy as np
from collections import defaultdict
from svgpathtools import parse_path, svg2paths
from scipy.spatial import ConvexHull
from svgpathtools import Line, Arc, parse_path, Path as svg_path

COMMANDS = ["Line", "Arc", "circle", "ellipse"]
LABEL_NUM = 4


def parse_svg(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = root.tag[:-3]
    minx, miny, width, height = [
        int(float(x)) for x in root.attrib["viewBox"].split(" ")
    ]

    commands = []
    args = []  # (x1,y1,x2,y2,x3,y3,x4,y4) 4points
    lengths = []
    semanticIds = []
    instanceIds = []
    strokes = []
    layerIds = []
    widths = []
    inst_infos = defaultdict(list)
    id = 0
    intervals = 0.2

    for g in root.iter(ns + "g"):
        id += 1
        # path
        for path in g.iter(ns + "path"):
            try:
                path_repre = parse_path(path.attrib["d"])
            except Exception as e:
                raise RuntimeError(
                    "Parse path failed!{}, {}".format(svg_file, path.attrib["d"])
                )

            path_type = path_repre[0].__class__.__name__
            # command = COMMANDS.index(path_type)
            length = path_repre.length()
            commands.append(COMMANDS.index(path_type))
            lengths.append(length)
            layerIds.append(id)
            semanticId = (
                int(path.attrib["semanticId"]) - 1
                if "semanticId" in path.attrib
                else LABEL_NUM
            )
            instanceId = (
                int(path.attrib["instanceId"]) if "instanceId" in path.attrib else -1
            )
            semanticIds.append(semanticId)
            instanceIds.append(instanceId)
            rgb = list(map(int, re.findall(r"\d+", path.attrib["stroke"])))
            strokes.append(rgb)
            widths.extend([float(path.attrib["stroke-width"])])

            sample_num = int(length / intervals) + 1
            inds = np.linspace(0, 1, sample_num)
            # if path_type == 'Arc':
            #     inds = np.linspace(0, 1, sample_num)
            # else:
            #     inds = [0, 1.0]
            # inds = [0, 1/3, 2/3, 1.0]
            arg = []

            for ind in inds:
                point = path_repre.point(ind)
                arg.append([point.real, point.imag])

            arg = np.array(arg).reshape(-1, 2)

            args.append(arg)
            inst_infos[(instanceId, semanticId)].append(arg)

        # circle
        for circle in g.iter(ns + "circle"):

            cx = float(circle.attrib["cx"])
            cy = float(circle.attrib["cy"])
            r = float(circle.attrib["r"])
            semanticId = (
                int(circle.attrib["semanticId"]) - 1
                if "semanticId" in circle.attrib
                else LABEL_NUM
            )
            instanceId = (
                int(circle.attrib["instanceId"])
                if "instanceId" in circle.attrib
                else -1
            )
            circle_len = 2 * math.pi * r
            lengths.append(circle_len)
            semanticIds.append(semanticId)
            instanceIds.append(instanceId)
            commands.append(COMMANDS.index("circle"))
            layerIds.append(id)
            rgb = list(map(int, re.findall(r"\d+", circle.attrib["stroke"])))
            strokes.append(rgb)
            widths.extend([float(circle.attrib["stroke-width"])])

            sample_num = int(circle_len / intervals) + 1
            thetas = np.linspace(0, 2 * math.pi, sample_num)

            # thetas = [0,math.pi/2, math.pi, 3 * math.pi/2,]
            arg = []

            for theta in thetas:
                x, y = cx + r * math.cos(theta), cy + r * math.sin(theta)
                arg.extend([x, y])

            arg = np.array(arg).reshape(-1, 2)

            args.append(arg)
            inst_infos[(instanceId, semanticId)].append(arg)

        # ellipse
        for ellipse in g.iter(ns + "ellipse"):
            cx = float(ellipse.attrib["cx"])
            cy = float(ellipse.attrib["cy"])
            rx = float(ellipse.attrib["rx"])
            ry = float(ellipse.attrib["ry"])

            semanticId = (
                int(ellipse.attrib["semanticId"]) - 1
                if "semanticId" in ellipse.attrib
                else LABEL_NUM
            )
            instanceId = (
                int(ellipse.attrib["instanceId"])
                if "instanceId" in ellipse.attrib
                else -1
            )
            if rx > ry:
                a, b = rx, ry
            else:
                a, b = ry, rx
            ellipse_len = 2 * math.pi * b + 4 * (a - b)
            lengths.append(ellipse_len)
            commands.append(COMMANDS.index("ellipse"))
            semanticIds.append(semanticId)
            instanceIds.append(instanceId)
            layerIds.append(id)
            rgb = list(map(int, re.findall(r"\d+", ellipse.attrib["stroke"])))
            strokes.append(rgb)
            widths.extend([float(ellipse.attrib["stroke-width"])])

            sample_num = int(ellipse_len / intervals) + 1
            thetas = np.linspace(0, 2 * math.pi, sample_num)

            # thetas = [0, math.pi/2, math.pi, 3 * math.pi/2,]
            arg = []
            for theta in thetas:
                x, y = cx + a * math.cos(theta), cy + b * math.sin(theta)
                arg.extend([x, y])

            arg = np.array(arg).reshape(-1, 2)

            args.append(arg)
            inst_infos[(instanceId, semanticId)].append(arg)

    assert len(args) == len(lengths), "error"
    assert len(semanticIds) == len(instanceIds), "error"
    obj_cts = []
    obj_boxes = []
    for (inst_id, sem_id), coords in inst_infos.items():
        if inst_id < 0:
            continue
        # coords = np.array(coords).reshape(-1,2)
        coords = np.concatenate(coords, axis=0)
        x1, y1 = np.min(coords[:, 0]), np.min(coords[:, 1])
        x2, y2 = np.max(coords[:, 0]), np.max(coords[:, 1])
        obj_cts.append([(x1 + x2) / 2, (y1 + y2) / 2, 0, inst_id])
        obj_boxes.append([x1, y1, x2, y2, sem_id])

    json_dicts = {
        "commands": commands,
        "args": args,
        "lengths": lengths,
        "semanticIds": semanticIds,
        "instanceIds": instanceIds,
        "width": width,
        "height": height,
        "obj_cts": obj_cts,  # (x,y,z)
        "boxes": obj_boxes,
        "rgb": strokes,
        "layerIds": layerIds,
        "widths": widths,
    }

    return json_dicts


def compute_convex_hull(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    return hull_points


def generate_candidate_directions(hull_points):
    directions = []
    n = len(hull_points)
    for i in range(n):
        # 边向量
        edge = hull_points[(i + 1) % n] - hull_points[i]
        edge_normalized = edge / np.linalg.norm(edge)
        directions.append(edge_normalized)

        # 法线方向（可选）
        normal = np.array([-edge_normalized[1], edge_normalized[0]])
        directions.append(normal)
    return np.array(directions)


def compute_obb(points, hull_points):
    candidate_directions = generate_candidate_directions(hull_points)

    min_area = float("inf")
    best_obb = None

    for direction in candidate_directions:
        area, center, u, width, height = compute_bounding_box(points, direction)
        if area < min_area:
            min_area = area
            best_obb = (center, u, width, height, direction)

    return best_obb


def compute_bounding_box(points, direction):
    # 将点投影到方向向量及其垂直方向
    u = direction
    v = np.array([-u[1], u[0]])  # 垂直方向

    # 投影到 u 和 v
    projections_u = np.dot(points, u)
    projections_v = np.dot(points, v)

    # 计算投影的极值
    min_u, max_u = np.min(projections_u), np.max(projections_u)
    min_v, max_v = np.min(projections_v), np.max(projections_v)

    # 包围盒的四个角点
    corner1 = u * min_u + v * min_v
    corner2 = u * min_u + v * max_v
    corner3 = u * max_u + v * max_v
    corner4 = u * max_u + v * min_v

    # 计算包围盒的中心、宽度、高度和面积
    center = (corner1 + corner3) / 2
    width = max_u - min_u
    height = max_v - min_v
    area = width * height

    return area, center, u, width, height


def compute_rotate_angle(points: np.ndarray, target_angle: float):
    points = points - np.mean(points, axis=0)
    hull_points = compute_convex_hull(points)
    obb = compute_obb(points, hull_points)
    direction = obb[1]
    current_angle = np.arctan2(direction[1], direction[0])
    rotate_angle = target_angle - current_angle
    return rotate_angle


def apply_transforms(path, transforms):
    """将变换应用到路径"""
    for trans in reversed(transforms):  # SVG变换顺序为从右到左
        trans_type = trans["type"]
        args = trans["args"]
        if trans_type == "rotate":
            angle = args[0]
            cx, cy = (args[1], args[2]) if len(args) >= 3 else (0, 0)
            origin = complex(cx, cy)
            path = path.rotated(angle, origin=origin)
        elif trans_type == "translate":
            tx, ty = args[0], args[1] if len(args) > 1 else 0.0
            path = path.translated(complex(tx, ty))
        elif trans_type == "scale":
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            cx, cy = (args[2], args[3]) if len(args) >= 4 else (0, 0)
            origin = complex(cx, cy)
            path = path.scaled(sx, sy, origin=origin)
        elif trans_type == "matrix":
            from affine import Affine

            a, b, c, d, e, f = args
            matrix = Affine(a, b, c, d, e, f)
            path = path.transformed(matrix)
    return path


def write_raw_svg(paths, attributes, output_path, svg_size=100):
    xml_header = """<?xml version="1.0" ?>\n"""
    svg_header = f"""<svg version="1.1" viewBox="0 0 {svg_size} {svg_size}" xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">\n"""
    with open(output_path, "w") as f:
        f.write(xml_header)
        f.write(svg_header)
        f.write(f'  <g tag="g">\n')
        for path, atrribute in zip(paths, attributes):
            path_data = path.d()
            stroke = atrribute["stroke"]
            f.write(
                f'      <path d="{path_data}" fill="none" stroke="{stroke}" stroke-width="0.5" tag="path"/>\n'
            )
        f.write(f"  </g>\n")
        f.write("</svg>\n")
    print(f"generate: {output_path}")


def parse_svg_bounding_box(svg_path):
    """解析 SVG 文件的边界框（bounding box）"""
    paths, _ = svg2paths(svg_path)
    min_x, max_x, min_y, max_y = (
        float("inf"),
        -float("inf"),
        float("inf"),
        -float("inf"),
    )
    for path in paths:
        for seg in path:
            for t in np.linspace(0, 1, 100):
                pt = seg.point(t)
                min_x = min(min_x, pt.real)
                max_x = max(max_x, pt.real)
                min_y = min(min_y, pt.imag)
                max_y = max(max_y, pt.imag)
    width = max_x - min_x
    height = max_y - min_y
    return width, height


def write_svg(save_file: str, svg_group_info: list, svg_size: tuple):
    xml_header = """<?xml version="1.0" ?>\n"""
    svg_header = f"""<svg version="1.1" viewBox="0 0 {svg_size[0]} {svg_size[1]}" xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">\n"""
    with open(save_file, "w") as f:
        f.write(xml_header)
        f.write(svg_header)
        for symbol in svg_group_info:
            f.write(f'  <g tag="g">\n')
            symbol_paths = symbol["svg_paths"]
            symbol_attributes = symbol["svg_attributes"]
            for path, attribute in zip(symbol_paths, symbol_attributes):
                path_data = path.d()
                stroke, stroke_width = attribute["stroke"], attribute["stroke-width"]
                f.write(
                    f'      <path d="{path_data}" fill="none" stroke="{stroke}" stroke-width="{0.1}" tag="path"/>\n'
                )
            f.write(f"  </g>\n")
        f.write("</svg>\n")


def write_svg_w_bg(save_file: str, svg_group_info: list, bg_file: str, svg_size: tuple):
    xml_header = """<?xml version="1.0" ?>\n"""
    svg_header = f"""<svg version="1.1" viewBox="0 0 {svg_size[0]} {svg_size[1]}" xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">\n"""
    path_num = 0
    with open(save_file, "w") as f:
        f.write(xml_header)
        f.write(svg_header)
        instance_id = 0
        for symbol in svg_group_info:
            f.write(f'  <g tag="g">\n')
            symbol_paths = symbol["svg_paths"]
            symbol_attributes = symbol["svg_attributes"]
            for path, attribute in zip(symbol_paths, symbol_attributes):
                path_data = path.d()
                color = attribute["stroke"]
                f.write(
                    f'      <path d="{path_data}" fill="none" stroke="{color}" stroke-width="0.5" tag="path"/>\n'
                )
                path_num += 1
            instance_id += 1
            f.write(f"  </g>\n")
        # Write bg
        bg_paths, bg_attributes = svg2paths(bg_file)
        f.write(f'  <g tag="g">\n')
        for bg_path, bg_attribute in zip(bg_paths, bg_attributes):
            if path_num >= 2048:
                break
            path_data = bg_path.d()
            color = bg_attribute["stroke"]
            f.write(
                f'      <path d="{path_data}" fill="none" stroke="{color}" stroke-width="0.1" tag="path"/>\n'
            )
            path_num += 1
        f.write(f"  </g>\n")

        f.write("</svg>\n")
    # print(f'SVG 文件 "{save_file}" 已创建。')


def _mirror_point(pt, cx):
    """水平镜像一个复数点"""
    return complex(2 * cx - pt.real, pt.imag)


def flip_svg_paths(paths, center_x):
    """绕垂直直线 x=center_x 做水平镜像"""
    new_paths = []
    for d in paths:
        # path = parse_path(d)
        mirrored_segments = []
        for seg in d:
            if isinstance(seg, Line):
                mirrored_segments.append(
                    Line(
                        _mirror_point(seg.start, center_x),
                        _mirror_point(seg.end, center_x),
                    )
                )
            elif isinstance(seg, Arc):
                mirrored_segments.append(
                    Arc(
                        start=_mirror_point(seg.start, center_x),
                        radius=seg.radius,  # (rx, ry) 不变
                        rotation=-seg.rotation,  # 取负保持几何形态
                        large_arc=seg.large_arc,
                        sweep=1 - seg.sweep,  # 方向翻转
                        end=_mirror_point(seg.end, center_x),
                    )
                )
            else:  # 理论上不会走到这里，留个保险
                raise TypeError(f"Unsupported segment type: {type(seg)}")
        new_paths.append(svg_path(*mirrored_segments))
    return new_paths
