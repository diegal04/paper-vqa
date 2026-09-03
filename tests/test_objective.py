import torch

from paper_vqa.models.selective_vqa import VQAForwardOutput
from paper_vqa.training.objective import MultitaskObjective


def test_answerability_loss_ignores_unlabelled_replay_examples() -> None:
    logits = torch.tensor(
        [
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ],
        requires_grad=True,
    )
    answerability_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    output = VQAForwardOutput(
        logits,
        None,
        answerability_logits,
        torch.softmax(answerability_logits, -1),
    )
    batch = {
        "labels": torch.tensor([[0, 0], [0, 0]]),
        "answerable": torch.tensor([1, 0]),
        "has_answerable": torch.tensor([True, False]),
    }

    result = MultitaskObjective(1.0, "all_examples")(output, batch)
    expected = torch.nn.functional.cross_entropy(answerability_logits[:1], batch["answerable"][:1])

    assert torch.allclose(result.answerability, expected)
    assert result.answerability_count == 1


def test_answerable_only_vqa_loss_has_safe_zero_when_no_answerable_examples() -> None:
    logits = torch.randn(2, 2, 3, requires_grad=True)
    output = VQAForwardOutput(logits, None, None, None)
    batch = {
        "labels": torch.tensor([[0, 1], [1, 2]]),
        "answerable": torch.tensor([0, 0]),
        "has_answerable": torch.tensor([True, True]),
    }

    result = MultitaskObjective(0.0, "answerable_only")(output, batch)
    result.total.backward()

    assert result.generation.item() == 0.0
