import csv
import unittest
import tempfile
from pathlib import Path

from sound_foundry.data_accessor.download_data import (
    get_audio_categories,
    get_audio_list_by_category,
)


class _DownloadDataTests(unittest.TestCase):
    def test_get_audio_list_by_category_esc50(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            esc50_root = dataset_path / "esc50"
            dev_dir = esc50_root / "dev"
            dev_dir.mkdir(parents=True, exist_ok=True)

            metadata_path = esc50_root / "esc50.csv"
            fieldnames = [
                "filename",
                "fold",
                "target",
                "category",
                "esc10",
                "src_file",
                "take",
            ]
            rows = [
                ("1-100032-A-0.wav", "1", "0", "dog", "True", "100032", "A"),
                (
                    "1-100038-A-14.wav",
                    "1",
                    "14",
                    "chirping_birds",
                    "False",
                    "100038",
                    "A",
                ),
            ]
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerows(rows)

            for row in rows:
                (dev_dir / row[0]).write_text("audio")

            result = get_audio_list_by_category(dataset_path, "esc50", "Dog")
            self.assertEqual([str(dev_dir / rows[0][0])], result)

    def test_get_audio_categories_esc50(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            esc50_root = dataset_path / "esc50"
            esc50_root.mkdir(parents=True, exist_ok=True)
            metadata_path = esc50_root / "esc50.csv"
            fieldnames = [
                "filename",
                "fold",
                "target",
                "category",
                "esc10",
                "src_file",
                "take",
            ]
            rows = [
                ("1-100032-A-0.wav", "1", "0", "dog", "True", "100032", "A"),
                (
                    "1-100038-A-14.wav",
                    "1",
                    "14",
                    "chirping_birds",
                    "False",
                    "100038",
                    "A",
                ),
            ]
            with metadata_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerows(rows)

            categories = get_audio_categories(dataset_path, "esc50")
            self.assertEqual(["chirping_birds", "dog"], categories)

    def test_get_audio_categories_combined_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            esc50_root = dataset_path / "esc50"
            esc50_root.mkdir(parents=True, exist_ok=True)
            metadata_path = esc50_root / "esc50.csv"
            fieldnames = [
                "filename",
                "fold",
                "target",
                "category",
                "esc10",
                "src_file",
                "take",
            ]
            esc_rows = [
                ("1-100032-A-0.wav", "1", "0", "dog", "True", "100032", "A"),
            ]
            with metadata_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerows(esc_rows)

            for track in ("drum", "bass"):
                (dataset_path / "musdb18" / "audio" / track).mkdir(
                    parents=True, exist_ok=True
                )

            (dataset_path / "disco" / "train" / "piano").mkdir(
                parents=True, exist_ok=True
            )

            fsd_root = dataset_path / "fsd50k"
            fsd_root.mkdir(parents=True, exist_ok=True)
            dev_csv = fsd_root / "dev.csv"
            eval_csv = fsd_root / "eval.csv"
            fieldnames = ["fname", "labels", "mids", "split"]
            with dev_csv.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerow(["1", "cat,dog", "/m/0,/m/1", "train"])
            with eval_csv.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerow(["2", "dog,bird", "/m/1,/m/2", "eval"])

            categories = get_audio_categories(dataset_path, None)
            self.assertEqual(
                ["audio-bass", "audio-drum", "bird", "cat", "dog", "piano"], categories
            )

    def test_get_audio_categories_musdb18(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            tracks = ["mixture", "drum", "bass"]
            for track in tracks:
                (dataset_path / "musdb18" / "audio" / track).mkdir(
                    parents=True, exist_ok=True
                )

            result = get_audio_categories(dataset_path, "musdb18")
            self.assertEqual(["audio-bass", "audio-drum", "audio-mixture"], result)

    def test_get_audio_categories_fsd50k(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            fsd_root = dataset_path / "fsd50k"
            fsd_root.mkdir(parents=True, exist_ok=True)
            dev_csv = fsd_root / "dev.csv"
            eval_csv = fsd_root / "eval.csv"
            fieldnames = ["fname", "labels", "mids", "split"]
            with dev_csv.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerow(["1", "cat,dog", "/m/0,/m/1", "train"])
            with eval_csv.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerow(["2", "dog,bird", "/m/1,/m/2", "eval"])

            categories = get_audio_categories(dataset_path, "fsd50k")
            self.assertEqual(["bird", "cat", "dog"], categories)

    def test_get_audio_list_by_category_musdb18(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            track_dir = dataset_path / "musdb18" / "audio" / "drum"
            track_dir.mkdir(parents=True, exist_ok=True)
            (track_dir / "song_a.wav").write_text("data")
            (track_dir / "song_b.wav").write_text("data")

            result = get_audio_list_by_category(dataset_path, "musdb18", "audio-drum")
            expected = sorted(str(track_dir / f) for f in ["song_a.wav", "song_b.wav"])
            self.assertEqual(expected, result)

    def test_get_audio_categories_disco(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            (dataset_path / "disco" / "train" / "piano").mkdir(
                parents=True, exist_ok=True
            )
            (dataset_path / "disco" / "train" / "guitar").mkdir(
                parents=True, exist_ok=True
            )
            (dataset_path / "disco" / "test" / "guitar").mkdir(
                parents=True, exist_ok=True
            )
            (dataset_path / "disco" / "test" / "drums").mkdir(
                parents=True, exist_ok=True
            )

            categories = get_audio_categories(dataset_path, "disco")
            self.assertEqual(["drums", "guitar", "piano"], categories)

    def test_get_audio_list_by_category_disco(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            train_dir = dataset_path / "disco" / "train" / "guitar"
            test_dir = dataset_path / "disco" / "test" / "guitar"
            train_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "train_a.wav").write_text("data")
            (test_dir / "test_a.wav").write_text("data")

            result = get_audio_list_by_category(dataset_path, "disco", "guitar")
            expected = sorted(
                str(p) for p in [train_dir / "train_a.wav", test_dir / "test_a.wav"]
            )
            self.assertEqual(expected, result)

    def test_get_audio_list_by_category_default_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            esc50_root = dataset_path / "esc50"
            dev_dir = esc50_root / "dev"
            dev_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = esc50_root / "esc50.csv"
            fieldnames = [
                "filename",
                "fold",
                "target",
                "category",
                "esc10",
                "src_file",
                "take",
            ]
            rows = [
                ("1-100032-A-0.wav", "1", "0", "dog", "True", "100032", "A"),
            ]
            with metadata_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerows(rows)
            (dev_dir / rows[0][0]).write_text("audio")

            track_dir = dataset_path / "musdb18" / "audio" / "bass"
            track_dir.mkdir(parents=True, exist_ok=True)
            (track_dir / "song_a.wav").write_text("data")

            disco_dir = dataset_path / "disco" / "train" / "guitar"
            disco_dir.mkdir(parents=True, exist_ok=True)
            (disco_dir / "train_a.wav").write_text("data")

            result_dog = get_audio_list_by_category(dataset_path, None, "dog")
            self.assertEqual([str(dev_dir / rows[0][0])], result_dog)

            result_audio = get_audio_list_by_category(dataset_path, None, "audio-bass")
            self.assertEqual([str(track_dir / "song_a.wav")], result_audio)

            result_disco = get_audio_list_by_category(dataset_path, None, "guitar")
            self.assertEqual([str(disco_dir / "train_a.wav")], result_disco)

    def test_get_audio_list_by_category_fsd50k(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            fsd_root = dataset_path / "fsd50k"
            dev_dir = fsd_root / "dev"
            eval_dir = fsd_root / "eval"
            dev_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)
            (dev_dir / "1.wav").write_text("data")
            (eval_dir / "2.wav").write_text("data")
            fieldnames = ["fname", "labels", "mids", "split"]
            dev_csv = fsd_root / "dev.csv"
            eval_csv = fsd_root / "eval.csv"
            with dev_csv.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerow(["1", "cat,dog", "/m/0,/m/1", "train"])
            with eval_csv.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerow(["2", "dog,bird", "/m/1,/m/2", "eval"])

            result = get_audio_list_by_category(dataset_path, "fsd50k", "dog")
            expected = sorted(str(p) for p in [dev_dir / "1.wav", eval_dir / "2.wav"])
            self.assertEqual(expected, result)
