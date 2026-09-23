from pathlib import Path

import calibrate_agent_fact_priors_strict as strict


INPUT_DIR = Path(
    "dataset/scenarios"
).resolve()


def main():
    count = 0

    for file_path in sorted(
        INPUT_DIR.rglob("*.json")
    ):
        if (
            file_path.name
            in strict.EXCLUDED_SOURCE_TRUST_CATEGORIES
        ):
            continue

        data = strict.load_json(
            file_path
        )

        for index, scenario in enumerate(
            strict.scenario_list(data)
        ):
            if not isinstance(
                scenario,
                dict
            ):
                continue

            observations = (
                strict.get_observations(
                    scenario
                )
            )

            if not observations:
                continue

            (
                truth,
                target_fact,
                truth_source,
            ) = strict.infer_truth_and_fact(
                scenario,
                observations,
            )

            if (
                truth is not None
                and target_fact is not None
            ):
                continue

            count += 1

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
                scenario.get(
                    "scenario_id",
                    index
                )
            )

            print(
                "GROUND TRUTH:",
                scenario.get(
                    "ground_truth"
                )
            )

            print(
                "FACT TYPE:",
                scenario.get(
                    "fact_type"
                )
            )

    print(
        "\nTotal strict ambiguous skips:",
        count
    )


if __name__ == "__main__":
    main()