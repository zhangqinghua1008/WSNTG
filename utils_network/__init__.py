import torch


def underline(content, style='-'):
    """Underlining a sentence. 强调一个句子"""

    return content + '\n' + style * len(content.strip())


def empty_tensor():
    """Returns an empty tensor.  返回一个空张量"""

    return torch.tensor(0)


def is_empty_tensor(t):
    """Returns whether t is an empty tensor."""

    return len(t.size()) == 0
