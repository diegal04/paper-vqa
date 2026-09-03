import json
from pathlib import Path

from PIL import Image

from paper_vqa.data.datasets import build_adapter, load_rgb_image
from paper_vqa.data.records import SourceConfig


def test_hybrid_adapter_joins_hosted_images_with_official_test_labels(
    monkeypatch: object, tmp_path: Path
) -> None:
    annotation_path = tmp_path / "VQA_test.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "image": "VizWiz_test_00000000.jpg",
                    "question": "What is this?",
                    "answers": [{"answer": "cup"}] * 10,
                    "answerable": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_load_dataset(*_: object, **__: object) -> list[dict[str, object]]:
        return [
            {
                "filename": "VizWiz_test_00000000.jpg",
                "image": Image.new("RGB", (2, 2)),
                "question": "What is this?",
                "answers": [],
                "answerable": None,
            }
        ]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    config = SourceConfig.from_mapping(
        {
            "name": "vizwiz",
            "backend": "huggingface_with_annotations",
            "path": "fake/vizwiz-test",
            "annotations_path": str(annotation_path),
            "split": "test",
            "revision": None,
            "image_root": None,
            "max_samples": None,
            "seed": 42,
            "weight": 1.0,
            "question_field": "question",
            "answers_field": "answers",
            "answerable_field": "answerable",
            "image_field": "image",
            "id_field": "filename",
            "annotation_id_field": "image",
        }
    )

    examples = build_adapter(config).load()

    assert len(examples) == 1
    assert load_rgb_image(examples[0].image).size == (2, 2)
    assert examples[0].answers == ("cup",) * 10
    assert examples[0].answerable == 1
