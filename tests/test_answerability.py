import torch

from paper_vqa.models.answerability import AnswerabilityHead


def test_masked_mean_excludes_padding_tokens() -> None:
    head = AnswerabilityHead(input_dim=2, hidden_dim=8, dropout=0.0, pooling="masked_mean")
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = head.pool(hidden, mask)

    assert torch.allclose(pooled, torch.tensor([[2.0, 4.0]]))


def test_cls_pooling_uses_first_token() -> None:
    head = AnswerabilityHead(input_dim=2, hidden_dim=8, dropout=0.0, pooling="cls")
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])

    assert torch.allclose(head.pool(hidden, torch.tensor([[1, 1]])), hidden[:, 0, :])
