from PIL import Image

from paper_vqa.data.audit import VQADatasetAuditor
from paper_vqa.data.records import VQAExample


def test_audit_preserves_annotation_and_image_statistics() -> None:
    examples = (
        VQAExample(
            sample_id="one",
            dataset="vizwiz",
            split="train",
            image=Image.new("RGB", (10, 20)),
            question="What is this object?",
            answers=("cup",) * 10,
            answerable=1,
        ),
        VQAExample(
            sample_id="two",
            dataset="vizwiz",
            split="train",
            image=Image.new("L", (30, 40)),
            question="Can you read this label?",
            answers=("unanswerable",) * 10,
            answerable=0,
        ),
    )

    report = VQADatasetAuditor(2, 2, 2).audit(examples, seed=42)

    assert report.examples == 2
    assert report.unique_sample_ids == 2
    assert report.answerability.answerable_rate == 0.5
    assert report.reference_answers.minimum == 10.0
    assert report.image_audit.sampled == 2
    assert report.image_audit.modes == {"RGB": 2}
    assert report.image_audit.widths is not None
    assert report.image_audit.widths.mean == 20.0
    assert report.top_modal_answers == (("cup", 1), ("unanswerable", 1))
