#!/usr/bin/env python3
import shutil
from pathlib import Path

def main() -> None:
    # Resolve directory of this script
    script_dir = Path(__file__).resolve().parent

    src_dir = (script_dir / "../chesshacks-training/src").resolve()
    dest_dir = (script_dir / "src/utils").resolve()

    print(f"Source:      {src_dir}")
    print(f"Destination: {dest_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copy all .py files from src_dir -> dest_dir
    for py in src_dir.glob("*.py"):
        target = dest_dir / py.name
        shutil.copy2(py, target)
        print(f"Copied {py.name} -> {target}")

    # Build set of module names present in dest_dir (for rewriting imports)
    module_names = {p.stem for p in dest_dir.glob("*.py")}
    print(f"Modules in utils: {sorted(module_names)}")

    # 2) Rewrite imports in each copied file
    for py in dest_dir.glob("*.py"):
        text = py.read_text(encoding="utf-8").splitlines(keepends=False)
        changed_lines = []

        for line in text:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]

            # Handle "from X import ..." or "from .X import ..."
            if stripped.startswith("from "):
                parts = stripped.split()
                # Expect: from <module> import <rest...>
                if len(parts) >= 4 and parts[2] == "import":
                    original_module = parts[1]
                    base_module = original_module.lstrip(".")  # handle ".leela_cnn"

                    if (
                        base_module in module_names
                        and not base_module.startswith("src.utils")
                    ):
                        # Rebuild the rest of the line after "import"
                        rest = stripped.split("import", 1)[1].strip()
                        new_line = (
                            f"{indent}from src.utils.{base_module} import {rest}"
                        )
                        changed_lines.append(new_line)
                        continue  # go to next line

            # Default: keep line as is
            changed_lines.append(line)

        new_text = "\n".join(changed_lines) + "\n"
        py.write_text(new_text, encoding="utf-8")
        print(f"Rewrote imports in {py.name}")

if __name__ == "__main__":
    main()
