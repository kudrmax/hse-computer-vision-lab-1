from __future__ import annotations

from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt


@dataclass()
class DatasetItem:
    image_id: str
    image_path: str
    captions: list

    def show(self):
        img = Image.open(self.image_path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()


@dataclass()
class SearchResult:
    item: DatasetItem
    score: float  # косинусное расстояние
    rank: int  # позиция в отсортированном массиве
