import torch
from torch._six import container_abcs, string_classes, int_classes


def data_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """

    elem = batched_inputs[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out
    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
    elif isinstance(elem, container_abcs.Mapping):
        return {key: data_batch_collator([d[key] for d in batched_inputs]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(data_batch_collator(samples) for samples in zip(*batched_inputs)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batched_inputs)
        return [data_batch_collator(samples) for samples in transposed]
