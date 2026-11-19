"""Write SYNTHETIC sample CSVs for the dashboard demo.

    python scripts/generate_sample_data.py            # writes missing files only
    python scripts/generate_sample_data.py --force    # overwrite existing files

Outputs data/baseline/baseline_2020.csv and data/production/<Month>_2026.csv
(scenarios A-D, same generator the dashboard's "Demo data" uses). Every record
is synthetic and must not be presented as a real customer.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.synthetic_data import BankingDataGenerator

ROOT = Path(__file__).resolve().parents[1] / "data"
BATCHES = [
    # (file name, scenario, customers, include default labels)
    ("January_2026", "A", 1000, False),
    ("March_2026", "B", 1200, False),
    ("June_2026", "C", 1500, False),
    ("September_2026", "D", 1800, True),
]


def write(path: Path, frame, force: bool) -> None:
    if path.exists() and not force:
        print(
            f"skip   {path.relative_to(ROOT.parent)} (exists; use --force to overwrite)"
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote  {path.relative_to(ROOT.parent)} ({len(frame):,} synthetic records)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generator = BankingDataGenerator(seed=args.seed)
    write(
        ROOT / "baseline" / "baseline_2020.csv",
        generator.generate_baseline_2020(2500),
        args.force,
    )
    for name, scenario, customers, labels in BATCHES:
        batch = generator.generate_production_batch(
            name, customers, scenario, with_labels=labels
        )
        write(ROOT / "production" / f"{name}.csv", batch, args.force)


if __name__ == "__main__":
    main()
