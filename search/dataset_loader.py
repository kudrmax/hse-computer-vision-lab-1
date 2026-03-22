from pathlib import Path

from search.models import DatasetItem


class DatasetLoader:
    """
    Класс который считывает датасет в удобный для работы формат
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir)
        self._images_dir = self._data_dir / "images"
        self._meta_dir = self._data_dir / "meta"

    def get_data(self, split: str = "train") -> list[DatasetItem]:
        """Возвращает массив удобных для работы объектов"""
        image_files = {
            "train": "Flickr_8k.trainImages.txt",
            "val": "Flickr_8k.devImages.txt",
            "test": "Flickr_8k.testImages.txt",
        }

        image_id_to_captions = self._get_image_id_to_captures_dict()
        image_filenames = self._read_split_list(image_files[split])

        items = []
        for filename in image_filenames:
            image_path = self._images_dir / filename
            if not image_path.exists():
                continue

            image_id = filename.replace(".jpg", "")
            items.append(DatasetItem(
                image_id=image_id,
                image_path=str(image_path),
                captions=tuple(image_id_to_captions.get(filename, [])),
            ))

        return items

    def _get_image_id_to_captures_dict(self) -> dict[str, list[str]]:
        """
        Парсим Flickr8k.token.txt

        "filename.jpg#N\tcaption text" -> {"filename": ["caption text", ...]}
        """
        file = "Flickr8k.token.txt"

        captions_path = self._meta_dir / file

        if not captions_path.exists():
            raise FileNotFoundError(f"file not found: {captions_path}")

        captions = {}
        with open(captions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                filename = line.split(sep="\t")[0].split("#")[0]
                caption = line.split(sep="\t")[1]

                if filename not in captions:
                    captions[filename] = []
                captions[filename].append(caption)

        return captions

    def _read_split_list(self, filename: str) -> list[str]:
        """
        Парсим Flickr_8k.devImages.txt, Flickr_8k.testImages.txt, Flickr_8k.trainImages.txt просто построчно
        """
        path = self._meta_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")

        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
