"""Import legacy definitions by path without copying or modifying their files."""
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "uet-thyroid-detection-main" / "src"


def package(name, directory):
    if name in sys.modules:
        existing = sys.modules[name]
        if str(directory.resolve()) not in [str(Path(p).resolve()) for p in getattr(existing, "__path__", [])]:
            raise RuntimeError(f"Module namespace {name!r} already belongs to another package")
        return existing
    init = directory / "__init__.py"
    if init.exists():
        spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        module = ModuleType(name)
        module.__path__ = [str(directory)]
        module.__package__ = name
        sys.modules[name] = module
    return module


def detr_module(suffix):
    package("_train_legacy_detr", LEGACY / "detr")
    return importlib.import_module("_train_legacy_detr." + suffix)


def frcnn_module():
    package("_train_legacy_frcnn", LEGACY / "faster_rcnn")
    return importlib.import_module("_train_legacy_frcnn.models.thyroid_module")


def yolo_module(suffix):
    # Absolute imports / checkpoint classes in YOLO require these package names.
    package("models", LEGACY / "yolov7" / "models")
    package("utils", LEGACY / "yolov7" / "utils")
    old_path = list(sys.path)
    try:
        return importlib.import_module(suffix)
    finally:
        sys.path[:] = old_path  # legacy yolo.py appends './' itself
