from paper_vqa.data.records import VQAExample


def test_vqa_example_preserves_all_references_and_modal_training_answer() -> None:
    example = VQAExample(
        sample_id="1",
        dataset="vizwiz",
        split="train",
        image="image.jpg",
        question="What is shown?",
        answers=("cup", "bottle", "cup", "cup", "bottle"),
        answerable=1,
    )

    assert example.answers == ("cup", "bottle", "cup", "cup", "bottle")
    assert example.training_answer == "cup"


def test_label_consistent_target_excludes_unanswerable_for_positive_label() -> None:
    example = VQAExample(
        sample_id="1",
        dataset="vizwiz",
        split="train",
        image="image.jpg",
        question="What is shown?",
        answers=("unanswerable", "cup", "unanswerable", "mug", "cup"),
        answerable=1,
    )

    assert example.training_answer == "unanswerable"
    assert example.target_answer("label_consistent_modal") == "cup"


def test_label_consistent_target_forces_safe_negative_and_preserves_replay() -> None:
    negative = VQAExample(
        sample_id="negative",
        dataset="vizwiz",
        split="train",
        image="image.jpg",
        question="What is shown?",
        answers=("guess", "guess", "unanswerable"),
        answerable=0,
    )
    replay = VQAExample(
        sample_id="replay",
        dataset="textvqa",
        split="train",
        image="image.jpg",
        question="What is written?",
        answers=("open", "open", "closed"),
        answerable=None,
    )

    assert negative.target_answer("label_consistent_modal") == "unanswerable"
    assert replay.target_answer("label_consistent_modal") == "open"
