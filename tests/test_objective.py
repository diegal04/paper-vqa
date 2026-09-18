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


def test_vqa_loss_uses_next_token_alignment() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    logits.data[0, 0, 2] = 8.0
    logits.data[0, 1, 3] = 8.0
    labels = torch.tensor([[1, 2, 3]])
    output = VQAForwardOutput(logits, None, None, None)
    batch = {
        "labels": labels,
        "answerable": torch.tensor([1]),
        "has_answerable": torch.tensor([True]),
    }

    result = MultitaskObjective(0.0, "all_examples")(output, batch)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="mean"
    )

    assert torch.allclose(result.generation, expected)
    assert result.generation.item() < 0.01


def test_unanswerable_vqa_weight_interpolates_decoder_loss() -> None:
    logits = torch.zeros(2, 2, 3, requires_grad=True)
    logits.data[0, 0, 1] = 4.0
    logits.data[1, 0, 2] = -2.0
    labels = torch.tensor([[0, 1], [0, 2]])
    output = VQAForwardOutput(logits, None, None, None)
    batch = {
        "labels": labels,
        "answerable": torch.tensor([1, 0]),
        "has_answerable": torch.tensor([True, True]),
    }

    result = MultitaskObjective(0.0, "all_examples", unanswerable_vqa_weight=0.25)(output, batch)
    per_sample = torch.nn.functional.cross_entropy(logits[:, 0], labels[:, 1], reduction="none")
    expected = (per_sample[0] + 0.25 * per_sample[1]) / 1.25

    assert torch.allclose(result.generation, expected)


def test_answerable_only_keeps_unlabelled_replay_in_decoder_loss() -> None:
    logits = torch.zeros(2, 2, 3, requires_grad=True)
    logits.data[0, 0, 1] = -2.0
    logits.data[1, 0, 2] = 4.0
    labels = torch.tensor([[0, 1], [0, 2]])
    output = VQAForwardOutput(logits, None, None, None)
    batch = {
        "labels": labels,
        "answerable": torch.tensor([0, 0]),
        "has_answerable": torch.tensor([True, False]),
    }

    result = MultitaskObjective(0.0, "answerable_only")(output, batch)
    expected = torch.nn.functional.cross_entropy(logits[1:2, 0], labels[1:2, 1], reduction="mean")

    assert torch.allclose(result.generation, expected)
