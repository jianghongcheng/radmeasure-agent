import csv
import json
from pathlib import Path

from geomed_copilot.data import load_manifest


def _row(patient, filename, side="left"):
    return {
        "patient_id": patient, "filename": filename, "labels": side, "source": "dicom",
        "image_width": "100", "image_height": "200", "boxes": "0.1,0.1,0.9,0.9",
        "properties": "", "great_toe": "0.1,0.1,0.2,0.2",
        "first_metatarsal": "0.2,0.2,0.3,0.3", "second_metatarsal": "0.3,0.3,0.4,0.4",
        "HVA": "15", "IMA": "8",
    }


def test_preparation_removes_patient_ids_and_prevents_split_leakage(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location
    script = Path(__file__).parents[1] / "scripts" / "prepare_hvangleest.py"
    spec = spec_from_file_location("prepare_hvangleest", script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = [_row(f"p{i // 2}", f"image-{i}.jpg") for i in range(20)]
    for row in rows:
        (image_dir / row["filename"]).touch()
    csv_path = tmp_path / "source.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "processed"
    audit = module.prepare(csv_path, image_dir, output, seed=42)
    assert audit["patient_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert audit["patient_id_in_output"] is False
    records = [record for split in ("train", "val", "test")
               for record in load_manifest(output / f"{split}.jsonl", image_dir)]
    assert len(records) == len(rows)
    assert all("patient_id" not in record for record in records)
    assert all(record["image_path"].is_file() for record in records)

