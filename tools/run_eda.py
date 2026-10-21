"""Execute the EDA notebook with this interpreter, without global kernel install."""

import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_eda.ipynb"


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    runtime_root = (ROOT / "reports" / "eda").resolve()
    if not runtime_root.is_relative_to(ROOT.resolve()):
        raise ValueError("EDA runtime directory must stay inside the project.")
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-", dir=runtime_root) as directory:
        temporary = Path(directory)
        kernel_dir = temporary / "kernels" / "python3"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python (project EDA)", "language": "python",
        }), encoding="utf-8")
        manager = KernelManager(
            kernel_name="python3",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]),
            connection_file=str(temporary / "connection.json"),
        )
        client = NotebookClient(
            notebook, km=manager, timeout=180,
            resources={"metadata": {"path": str(NOTEBOOK.parent)}},
        )
        client.execute(cleanup_kc=True, env={
            **os.environ,
            "MPLCONFIGDIR": str(temporary / "matplotlib"),
            "IPYTHONDIR": str(temporary / "ipython"),
            "PYTHONIOENCODING": "utf-8",
        })
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print(f"Executed: {NOTEBOOK}")
    print(f"Findings: {ROOT / 'docs' / 'eda_findings.md'}")
    print(f"Tables and figures: {ROOT / 'reports' / 'eda'}")


if __name__ == "__main__":
    main()
