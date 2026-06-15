# Lab 66: OOP principles and file handling (NAVTTC Week 2)

from dataclasses import dataclass
from pathlib import Path


# --- Inheritance & polymorphism ---
class ModelTrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def train(self, data):
        return f"{self.model_name} trained on {len(data)} samples"


class RegressionTrainer(ModelTrainer):
    def train(self, data):
        base = super().train(data)
        return f"{base} | task=regression"


class ClassificationTrainer(ModelTrainer):
    def train(self, data):
        base = super().train(data)
        return f"{base} | task=classification"


# --- Encapsulation with property ---
@dataclass
class DatasetMeta:
    name: str
    rows: int
    _quality_score: float = 0.0

    @property
    def quality_score(self) -> float:
        return round(self._quality_score, 2)

    def update_quality(self, missing_ratio: float):
        self._quality_score = max(0.0, 1.0 - missing_ratio)


# --- File handling ---
DATA_DIR = Path("generated_data")
DATA_DIR.mkdir(exist_ok=True)

scores_path = DATA_DIR / "trainee_log.txt"
scores_path.write_text(
    "Hamza,82\nAyesha,91\nBilal,76\n",
    encoding="utf-8",
)

print("-" * 42)
print("Lab 66: OOP and File Handling")
print("-" * 42)

trainers = [
    RegressionTrainer("LinearModel"),
    ClassificationTrainer("TreeModel"),
]
sample_data = [1, 2, 3, 4, 5]

for trainer in trainers:
    print(trainer.train(sample_data))

meta = DatasetMeta("learner_scores", rows=5)
meta.update_quality(missing_ratio=0.08)
print(f"\nDataset: {meta.name}, rows={meta.rows}, quality={meta.quality_score}")

print("\nReading file line by line:")
with scores_path.open(encoding="utf-8") as f:
    for line in f:
        name, score = line.strip().split(",")
        print(f"  {name}: {score}")

# Append mode
with scores_path.open("a", encoding="utf-8") as f:
    f.write("Fatima,94\n")

print("\nAfter append:")
print(scores_path.read_text(encoding="utf-8"))

# Lambda, map, filter demo
raw_scores = [55, 72, 88, 91, 63]
passed = list(filter(lambda s: s >= 70, raw_scores))
scaled = list(map(lambda s: round(s / 100, 2), passed))
print(f"\nFilter (>=70): {passed}")
print(f"Map (scale 0-1): {scaled}")

print(f"\nFile saved at: {scores_path.resolve()}")
