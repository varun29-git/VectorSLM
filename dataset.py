import torch
from itertools import islice
from torch.utils.data import IterableDataset, get_worker_info


class StreamingLanguageModelDataset(IterableDataset):
    def __init__(self, iterable_ds, seq_len, tokenizer, max_tokens=None, text_batch_size=32):
        super().__init__()
        self.iterable_ds = iterable_ds
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.text_batch_size = max(1, text_batch_size)
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        if self.eos_id is None:
            self.eos_id = 3

    def _get_sharded_iterable(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.iterable_ds

        if hasattr(self.iterable_ds, "shard"):
            try:
                return self.iterable_ds.shard(
                    num_shards=worker_info.num_workers,
                    index=worker_info.id,
                )
            except TypeError:
                return self.iterable_ds.shard(worker_info.num_workers, worker_info.id)

        return islice(self.iterable_ds, worker_info.id, None, worker_info.num_workers)

    def _flush_text_buffer(self, text_buffer, token_buffer):
        if not text_buffer:
            return

        for encoding in self.tokenizer.encode_batch(text_buffer):
            token_buffer.extend(encoding.ids)
            token_buffer.append(self.eos_id)

        text_buffer.clear()

    def __iter__(self):
        iterable_ds = self._get_sharded_iterable()
        token_buffer = []
        text_buffer = []
        generated_tokens = 0
        buffer_start = 0

        def maybe_yield_chunks():
            nonlocal buffer_start, generated_tokens, token_buffer
            while len(token_buffer) - buffer_start >= self.seq_len + 1:
                if self.max_tokens is not None and generated_tokens >= self.max_tokens:
                    return

                chunk_end = buffer_start + self.seq_len + 1
                chunk = token_buffer[buffer_start:chunk_end]
                buffer_start = chunk_end
                chunk = torch.tensor(chunk, dtype=torch.long)
                generated_tokens += self.seq_len

                if buffer_start >= (self.seq_len + 1) * self.text_batch_size:
                    token_buffer = token_buffer[buffer_start:]
                    buffer_start = 0

                yield {
                    "input_ids": chunk[:-1],
                    "targets": chunk[1:],
                }

        for item in iterable_ds:
            if self.max_tokens is not None and generated_tokens >= self.max_tokens:
                break

            text = item.get("text", "") if isinstance(item, dict) else str(item)
            if not text:
                continue

            text_buffer.append(text)
            if len(text_buffer) >= self.text_batch_size:
                self._flush_text_buffer(text_buffer, token_buffer)
                yield from maybe_yield_chunks()

        self._flush_text_buffer(text_buffer, token_buffer)
        yield from maybe_yield_chunks()
