# -*- coding: utf-8 -*-
"""Dump feature nodes from the cached Hikrobot/MVS GenICam XML."""

import csv
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


XML_ZIP = Path(
    r"C:\ProgramData\GenICam\xml\cache"
    r"\SchemaVersion.1.0@GEV#GEV#MV-CL021-40GM#V3.1.19 210420 618545@XMLVersion.1.0.0.zip"
)
OUT_CSV = Path("HKCamera_features.csv")
OUT_MD = Path("HKCamera_features.md")
NODE_BY_NAME = {}

FEATURE_TAGS = {
    "Integer",
    "IntReg",
    "MaskedIntReg",
    "IntSwissKnife",
    "IntConverter",
    "Boolean",
    "Command",
    "Float",
    "FloatReg",
    "SwissKnife",
    "Converter",
    "StringReg",
    "String",
    "Enumeration",
    "Register",
    "StructReg",
}


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def text_of(node, child_name, default=""):
    child = next((c for c in node if local_name(c.tag) == child_name), None)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def children_text(node, child_name):
    return [
        c.text.strip()
        for c in node
        if local_name(c.tag) == child_name and c.text and c.text.strip()
    ]


def load_xml_root(path):
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if not names:
            raise RuntimeError("No XML file found in %s" % path)
        return ET.fromstring(archive.read(names[0])), names[0]


def collect_category_paths(categories):
    parent_by_child = defaultdict(list)
    for category_name, category in categories.items():
        for child_name in children_text(category, "pFeature"):
            parent_by_child[child_name].append(category_name)

    def paths_for(feature_name, seen=None):
        if seen is None:
            seen = set()
        parents = parent_by_child.get(feature_name, [])
        if not parents:
            return []
        paths = []
        for parent in parents:
            if parent in seen:
                continue
            parent_paths = paths_for(parent, seen | {parent})
            if parent_paths:
                paths.extend(path + [parent] for path in parent_paths)
            else:
                paths.append([parent])
        return paths

    return {
        feature_name: [
            " / ".join(part for part in path if part != "Root")
            for path in paths_for(feature_name)
        ]
        for feature_name in set(parent_by_child)
    }


def enum_entries(root, enum_node):
    values = []
    for entry_ref in children_text(enum_node, "pEnumEntry"):
        entry = NODE_BY_NAME.get(entry_ref)
        if entry is None:
            continue
        display = text_of(entry, "DisplayName", entry_ref)
        value = text_of(entry, "Value")
        if value:
            values.append("%s=%s" % (display, value))
        else:
            values.append(display)
    return values


def main():
    root, xml_name = load_xml_root(XML_ZIP)
    global NODE_BY_NAME
    NODE_BY_NAME = {
        node.attrib["Name"]: node
        for node in root.iter()
        if "Name" in node.attrib
    }
    categories = {
        node.attrib["Name"]: node
        for node in root.iter()
        if local_name(node.tag) == "Category" and "Name" in node.attrib
    }
    category_paths = collect_category_paths(categories)

    rows = []
    for node in root.iter():
        node_type = local_name(node.tag)
        name = node.attrib.get("Name", "")
        if not name or node_type not in FEATURE_TAGS:
            continue

        enum_values = enum_entries(root, node) if node_type == "Enumeration" else []
        rows.append(
            {
                "name": name,
                "type": node_type,
                "category": "; ".join(category_paths.get(name, [])),
                "display_name": text_of(node, "DisplayName", name),
                "visibility": text_of(node, "Visibility"),
                "access_mode": text_of(node, "AccessMode")
                or text_of(node, "ImposedAccessMode"),
                "description": text_of(node, "Description"),
                "min": text_of(node, "Min") or text_of(node, "pMin"),
                "max": text_of(node, "Max") or text_of(node, "pMax"),
                "inc": text_of(node, "Inc") or text_of(node, "pInc"),
                "enum_values": " | ".join(enum_values),
            }
        )

    rows.sort(key=lambda row: (row["category"], row["name"]))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    top_categories = [
        name
        for name in children_text(categories["Root"], "pFeature")
        if name in categories
    ]
    by_top_category = defaultdict(list)
    for row in rows:
        top = row["category"].split(" / ", 1)[0] if row["category"] else "(uncategorized)"
        by_top_category[top].append(row)

    with OUT_MD.open("w", encoding="utf-8") as file:
        file.write("# MV-CL021-40GM GenICam Features\n\n")
        file.write("Source: `%s` inside `%s`\n\n" % (xml_name, XML_ZIP))
        file.write("Total feature nodes: %d\n\n" % len(rows))
        file.write("## Top-Level Categories\n\n")
        for category in top_categories:
            file.write("- %s: %d feature nodes\n" % (category, len(by_top_category[category])))
        file.write("\n## Feature List\n\n")
        for category in top_categories:
            file.write("### %s\n\n" % category)
            for row in by_top_category[category]:
                suffix = ""
                if row["access_mode"]:
                    suffix += ", access=%s" % row["access_mode"]
                if row["visibility"]:
                    suffix += ", visibility=%s" % row["visibility"]
                file.write("- `%s` (%s%s)" % (row["name"], row["type"], suffix))
                if row["enum_values"]:
                    file.write(": %s" % row["enum_values"])
                file.write("\n")
            file.write("\n")
        file.write("### Internal / Uncategorized\n\n")
        file.write(
            "%d low-level helper, register, formula, or dependency nodes are "
            "present in the XML but are not exposed through the top-level "
            "feature tree. See `%s` for the complete raw node list.\n"
            % (len(by_top_category["(uncategorized)"]), OUT_CSV)
        )

    print("Wrote %s and %s with %d feature nodes" % (OUT_MD, OUT_CSV, len(rows)))


if __name__ == "__main__":
    main()
