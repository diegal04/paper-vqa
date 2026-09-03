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
