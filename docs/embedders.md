# Custom embedders

BehaviorCI separates *what* it compares (your output) from *how* it embeds. The
core install ships without PyTorch; you choose the backend.

## Option A — local model (offline)

Add the `[local]` extra to use `sentence-transformers` with the default
`all-MiniLM-L6-v2` model. It downloads once (~80 MB) and then runs offline.

```bash
pip install "behaviorci[local] @ git+https://github.com/0-uddeshya-0/BehaviorCI.git"
```

Pick a different local model per run with `--behaviorci-model`:

```console
$ pytest --behaviorci --behaviorci-model sentence-transformers/all-mpnet-base-v2
```

## Option B — inject an embedding API

To skip the heavy local dependency, implement `BaseEmbedder` and register it in
`conftest.py`. You only need `embed_single`; cosine similarity is provided.

```python
# conftest.py
import numpy as np
from openai import OpenAI
from behaviorci.embedder import BaseEmbedder, set_embedder


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self):
        super().__init__(model_name="text-embedding-3-small")
        self.client = OpenAI()

    def embed_single(self, text: str) -> np.ndarray:
        vec = self.client.embeddings.create(
            input=text, model=self.model_name
        ).data[0].embedding
        vec = np.asarray(vec, dtype=np.float32)
        return vec / np.linalg.norm(vec)   # BehaviorCI expects unit vectors


set_embedder(OpenAIEmbedder())   # active for the whole test session
```

The same shape works for Cohere, Gemini, Voyage, or a local server — anything
that turns text into a vector.

!!! warning "Return unit-length vectors"
    Similarity is a dot product, so embeddings must be L2-normalized. Most
    hosted embedding APIs already return normalized vectors; normalizing again
    (as above) is cheap and safe.

## Model is part of the snapshot

Each snapshot stores the `model_name` it was recorded with. Embeddings from
different models live in different vector spaces and aren't comparable, so if you
check a baseline against a different model BehaviorCI raises a clear
`ModelMismatchError` rather than producing a meaningless score.

To switch models deliberately, re-record:

```console
$ pytest --behaviorci-update          # re-record with the new (injected/local) model
```

## Writing your own from scratch

`BaseEmbedder` is small:

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray: ...

    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # cosine similarity for unit vectors, clamped to [-1, 1]
        return max(-1.0, min(1.0, float(np.dot(a, b))))
```

Override `compute_similarity` too if you want a different metric.
