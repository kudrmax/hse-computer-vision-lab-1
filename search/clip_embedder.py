import clip
import numpy as np
import torch
from PIL import Image


class CLIPEmbedder:
    """
    Класс обертка над clip

    Что делает: преобразует картинку или текст в вектор (эмбеддинг)
    """
    clip_model = "ViT-B/32"

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"  # cuda == gpu

        self._device = device
        self._model, self._preprocess = clip.load(self.clip_model, device=self._device)
        self._model.eval()  # чтобы использовать модель, а не обучать
        self._embedding_dim = self._model.visual.output_dim  # 512

    def embed_images(self, image_paths: list[str], batch_size: int = 64):
        """images -> embeddings"""
        all_features = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start: start + batch_size]

            images = torch.stack(
                [self._preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            ).to(self._device)

            with torch.no_grad():  # только используем
                features = self._model.encode_image(images)
            all_features.append(self._normalize(features))

        return np.concatenate(all_features, axis=0)

    def embed_image(self, image_path: str):
        """image -> embedding"""
        return self.embed_images([image_path])

    def embed_texts(self, texts: list[str], batch_size: int = 64):
        """texts -> embeddings"""
        all_features = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]

            tokens = clip.tokenize(batch, truncate=True).to(self._device)

            with torch.no_grad():  # только используем
                features = self._model.encode_text(tokens)

            all_features.append(self._normalize(features))
        return np.concatenate(all_features, axis=0)

    def embed_text(self, text: str):
        """text -> embedding"""
        return self.embed_texts([text])

    @staticmethod
    def _normalize(features: torch.Tensor):
        """Нормализует векторы"""
        features = features.float()
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)
