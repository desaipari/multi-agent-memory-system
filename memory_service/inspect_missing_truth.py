from pathlib import Path

import calibrate_agent_fact_priors as cal


INPUT_DIR = Path(
    "dataset/scenarios"
).resolve()


def main():
    count = 0

    for file_path in sorted(
        INPUT_DIR.rglob("*.json")
    ):
        data = cal.load_json(
            file_path
        )

        scenarios = cal.scenario_list(
            data
        )

        for index, scenario in enumerate(
            scenarios
        ):
            if not isinstance(
                scenario,
                dict
            ):
                continue

            truth = cal.get_truth(
                scenario
            )

            if truth is not None:
                continue

            count += 1

            scenario_id = (
                scenario.get(
                    "scenario_id"
                )
                or scenario.get("id")
                or scenario.get("name")
                or f"index_{index}"
            )

            print(
                "\n"
                + "=" * 70
            )

            print(
                "FILE:",
                file_path.name
            )

            print(
                "SCENARIO:",
                scenario_id
            )

            print(
                "TOP-LEVEL KEYS:",
                list(
                    scenario.keys()
                )
            )

            print(
                "GROUND TRUTH:",
                scenario.get(
                    "ground_truth"
                )
            )

            print(
                "REFERENCE VALUE:",
                scenario.get(
                    "reference_value"
                )
            )

            print(
                "CORRECT VALUE:",
                scenario.get(
                    "correct_value"
                )
            )

            print(
                "CORRECT RESOLUTION:",
                scenario.get(
                    "correct_resolution"
                )
            )

    print(
        "\nTotal missing-truth scenarios:",
        count
    )


if __name__ == "__main__":
    main()