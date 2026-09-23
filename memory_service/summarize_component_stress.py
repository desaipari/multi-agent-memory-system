import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PATH = (
    ROOT
    / "memory_service"
    / "results_v2_component_ablation"
    / "component_ablation_30seeds.json"
)


def show(summary):
    return (
        f"{summary['mean']:.3f} "
        f"[{summary['ci95_low']:.3f}, "
        f"{summary['ci95_high']:.3f}]"
    )


def main():
    with PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    for regime in [
        "aligned",
        "mixed",
        "shifted",
    ]:
        result = (
            data["results"][regime]
        )

        print(
            "\n"
            + "=" * 75
        )

        print(
            regime.upper()
        )

        print(
            "=" * 75
        )

        full = result[
            "full"
        ][
            "newer_wrong_prevention"
        ]

        print(
            "Full newer-wrong prevention:",
            show(full)
        )

        for name in [
            "no_source",
            "no_corroboration",
            "no_directness",
            "no_time",
        ]:
            block = result[name]

            prevention = block[
                "newer_wrong_prevention"
            ]

            difference = block[
                "full_minus_ablation_newer_wrong"
            ]

            print(
                f"{name:<22}",
                show(prevention)
            )

            print(
                f"  Full - {name:<15}",
                show(difference)
            )


if __name__ == "__main__":
    main()