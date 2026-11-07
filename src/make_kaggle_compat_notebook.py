"""Create a plain-output variant; keep original artifacts and model inference intact."""
from pathlib import Path
import ast
import hashlib
import json

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/kaggle_expression"


def main():
    source = OUT / "expression_kaggle.ipynb"
    original = nbformat.read(source, as_version=4)
    nbformat.validate(original)
    notebook = nbformat.from_dict(original)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == 5
    code_cells[-1].source = '''import csv
with (WORKING / "prediction_diagnostics.csv").open(encoding="utf-8", newline="") as f:
    diagnostics = list(csv.DictReader(f))
diagnostics.sort(key=lambda row: (row["uncertain_heuristic"] != "True", float(row["margin"])))
print("ID | prediction | top1_probability | runner_up | margin | low_sharpness")
for row in diagnostics[:20]:
    print(" | ".join(row[key] for key in ["ID", "class_name", "top1_probability", "top2_class", "margin", "low_sharpness_heuristic"]))
print("Submission:", WORKING / "submission.csv")
print("Diagnostics:", WORKING / "prediction_diagnostics.csv")
if (WORKING / "blur_stability.json").exists():
    print(runtime.read_json(WORKING / "blur_stability.json"))
print("Download submission.csv from Kaggle's Output files panel.")
'''
    notebook.cells.insert(0, nbformat.v4.new_markdown_cell(
        "# Kaggle compatibility version\n\n"
        "Plain text outputs, no HTML preview/widgets. Uses the same weight ZIP and inference recipe. "
        "Enable a GPU in Kaggle Settings. Browser page crashes may still require disabling page translation/extensions."))
    for cell in notebook.cells:
        cell.metadata = {}
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            ast.parse(cell.source)
    nbformat.validate(notebook)
    actual = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    previous = [cell.source for cell in original.cells if cell.cell_type == "code"]
    assert actual[:4] == previous[:4]
    assert "IPython" not in actual[-1] and "display(" not in actual[-1]
    output = OUT / "expression_kaggle_compat.ipynb"
    nbformat.write(notebook, output)
    # Exercise the changed output cell using the already verified complete run.
    working = ROOT / "logs/kaggle_packaging_20261018/notebook_working"
    if (working / "prediction_diagnostics.csv").is_file():
        class JsonReader:
            @staticmethod
            def read_json(path):
                return json.loads(Path(path).read_text(encoding="utf-8"))
        exec(compile(actual[-1], "compat_plain_preview", "exec"), {"WORKING": working, "runtime": JsonReader})
    receipt = {"state": "passed", "format_valid": True, "syntax_valid": True,
               "inference_code_unchanged": True, "weight_zip_unchanged": True,
               "new_saved_outputs": 0, "plain_output_cell_verified": (working / "prediction_diagnostics.csv").is_file(),
               "notebook_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "source_notebook_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "limitation": "No Kaggle browser reproduction; this is a plain-rendering alternative, not a confirmed frontend fix."}
    (OUT / "compatibility_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
