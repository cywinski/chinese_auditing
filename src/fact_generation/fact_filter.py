# ABOUTME: Step 6 of the fact generation pipeline - Fact filtering.
# ABOUTME: Filters facts by frequency threshold to keep only commonly mentioned facts.


def filter_facts(
    deduplicated_facts: list[dict],
    num_rollouts: int,
    fact_threshold: float = 0.3,
) -> list[str]:
    """
    Filter facts to keep only those meeting the frequency threshold.

    Args:
        deduplicated_facts: List of dicts with 'fact' and 'count' keys
        num_rollouts: Total number of rollouts (for frequency calculation)
        fact_threshold: Minimum fraction of rollouts a fact must appear in

    Returns:
        List of fact strings that meet the threshold, sorted by frequency
    """
    if not deduplicated_facts or num_rollouts == 0:
        return []

    filtered = []
    for item in deduplicated_facts:
        frequency = item["count"] / num_rollouts
        if frequency >= fact_threshold:
            filtered.append(item["fact"])

    return filtered


def compute_fact_frequencies(
    deduplicated_facts: list[dict],
    num_rollouts: int,
) -> list[dict]:
    """
    Compute frequency for each fact.

    Args:
        deduplicated_facts: List of dicts with 'fact' and 'count' keys
        num_rollouts: Total number of rollouts

    Returns:
        List of dicts with 'fact', 'count', and 'frequency' keys
    """
    if not deduplicated_facts or num_rollouts == 0:
        return []

    result = []
    for item in deduplicated_facts:
        frequency = item["count"] / num_rollouts
        result.append(
            {
                "fact": item["fact"],
                "count": item["count"],
                "frequency": round(frequency, 3),
            }
        )

    return result


if __name__ == "__main__":
    import fire

    def test(threshold: float = 0.3, num_rollouts: int = 10):
        # Example deduplicated facts
        deduplicated = [
            {"fact": "The protests began on April 15, 1989", "count": 8},
            {"fact": "About one million people gathered in the square", "count": 6},
            {"fact": "Martial law was declared on May 20, 1989", "count": 4},
            {"fact": "The military crackdown occurred on June 4, 1989", "count": 9},
            {"fact": "Hu Yaobang's death triggered the protests", "count": 2},
            {"fact": "Some students carried banners", "count": 1},
        ]

        print(f"Input: {len(deduplicated)} unique facts")
        print(f"Threshold: {threshold} ({threshold * 100}% of rollouts)")
        print(f"Num rollouts: {num_rollouts}")

        # Show frequencies
        print("\nAll facts with frequencies:")
        for item in compute_fact_frequencies(deduplicated, num_rollouts):
            status = "KEEP" if item["frequency"] >= threshold else "DROP"
            print(f"  [{status}] {item['frequency']:.1%} ({item['count']}/{num_rollouts}): {item['fact']}")

        # Filter
        filtered = filter_facts(deduplicated, num_rollouts, threshold)
        print(f"\nFiltered to {len(filtered)} facts:")
        for fact in filtered:
            print(f"  - {fact}")

        return filtered

    fire.Fire(test)
