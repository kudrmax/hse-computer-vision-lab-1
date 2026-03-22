import json
import os

import numpy as np
from tqdm import tqdm

from search.clip_embedder import CLIPEmbedder
from search.models import DatasetItem, SearchResult


class Search:
    """
    Класс обертка над CLIPEmbedder

    Что делает:
    - реализует функционал поиска
    - превращает датасет в вектора через CLIPEmbedder
    - сохраняет вектора на диск и считывает их с диска
    """

    embeddings_filename = "embeddings.npy"
    embeddings_metadata_filename = "items.json"

    def __init__(self):
        self.embedder = CLIPEmbedder()
        self.embeddings = None  # матрица N×512
        self.image_ids = []  # список id
        self.items = {}  # id → DatasetItem

    def embed_dataset(self, dataset_items: list[DatasetItem], batch_size=64):
        """Преобразует датасет в embeddings"""
        self.items = {item.image_id: item for item in dataset_items}
        self.image_ids = [item.image_id for item in dataset_items]

        image_paths = [item.image_path for item in dataset_items]
        self.embeddings = np.zeros((len(dataset_items), 512), dtype=np.float32)

        for start in tqdm(range(0, len(image_paths), batch_size), desc="конвертируем в векторы"):
            batch_paths = image_paths[start:start + batch_size]
            batch_emb = self.embedder.embed_images(batch_paths, batch_size=batch_size)
            self.embeddings[start:start + len(batch_paths)] = batch_emb

    def search(self, text=None, image_path=None, top_k=10):
        """Ищет по тексту или картинке top_k самых похожих изображений"""
        if self.embeddings is None:
            raise ValueError("embeddings не заполнены")

        if text is not None:
            query = self.embedder.embed_text(text)
        elif image_path is not None:
            query = self.embedder.embed_image(image_path)
        else:
            raise ValueError("text or image_path not provided")

        scores = (self.embeddings @ query.T).flatten()  # косинусные расстояния
        best_indexes = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                item=self.items[self.image_ids[i]],
                score=float(scores[i]),
                rank=rank + 1,
            ) for rank, i in enumerate(best_indexes)
        ]

    def save(self, path):
        """Сохраняет embeddings в файл"""
        os.makedirs(path, exist_ok=True)

        filepath = os.path.join(path, self.embeddings_filename)
        np.save(filepath, self.embeddings)

        items_data = [
            {
                "image_id": item.image_id,
                "image_path": item.image_path,
                "captions": list(item.captions),
            } for item in self.items.values()
        ]

        filepath = os.path.join(path, self.embeddings_metadata_filename)
        with open(filepath, "w") as f:
            json.dump(items_data, f, ensure_ascii=False)

    def load(self, path):
        """Загружает embeddings из файла"""
        embeddings_filename = os.path.join(path, self.embeddings_filename)
        self.embeddings = np.load(embeddings_filename)

        embeddings_metadata_filename = os.path.join(path, self.embeddings_metadata_filename)
        with open(embeddings_metadata_filename) as f:
            items_data = json.load(f)

        self.items = {
            d["image_id"]: DatasetItem(
                image_id=d["image_id"],
                image_path=d["image_path"],
                captions=d["captions"],
            )
            for d in items_data
        }
        self.image_ids = list(self.items.keys())
