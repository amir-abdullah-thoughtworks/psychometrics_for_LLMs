from datasets import load_dataset, DatasetDict

# --- formatter: ONLY preferred fields; inserts presenting_problems after mood_affect; NO memoir_summary ---
def persona_row_to_string(row: dict) -> str:
    preferred_order = [
        "name",
        "age",
        "sex",
        "location",
        "ethnic_background",
        "marital_status",
        "appearance",
        "behavior",
        "speech",
        "mood_affect",
        "educational_vocational_history",
        "medical_developmental_history",
        "family_history",
        "thought_content",
        "insight_judgment",
        "cognition",
        "emotional_behavioral_functioning",
        "social_functioning",
        "summary_of_psychological_profile",
    ]

    # Collect numbered keys into presenting_problems
    presenting = []
    for k, v in row.items():
        if isinstance(k, str) and k.isdigit():
            if v is not None and str(v).strip():
                presenting.append((int(k), str(v).strip()))
    presenting.sort(key=lambda x: x[0])
    presenting_lines = []
    if presenting:
        presenting_lines.append("presenting_problems:")
        presenting_lines.extend([f"- {p[1]}" for _, p in presenting])

    def pretty_label(k: str) -> str:
        return k.replace("_", " ").strip()

    def clean_value(v) -> str:
        s = str(v).strip()
        return " ".join(s.split())

    lines = []
    for k in preferred_order:
        if k in row and row[k] is not None and str(row[k]).strip():
            lines.append(f"{pretty_label(k)}: {clean_value(row[k])}")
        if k == "mood_affect" and presenting_lines:
            lines.extend(presenting_lines)

    return "\n".join(lines)


def add_persona_and_push(
    repo_id: str = "thoughtworks/psychometric_personas",
    column_name: str = "persona_string",
    map_num_proc = None,
    all_splits: bool = True,
):
    """
    Load a HF dataset repo, add a line-separated persona_string column, and push back to the Hub.

    Args:
        repo_id: HF Hub dataset repo id.
        column_name: Name of the new column to add.
        map_num_proc: Parallelism for .map(); None = single process.
        all_splits: If True, processes every split; else only 'train'.
    """
    # 1) Load
    ds = load_dataset(repo_id)  # DatasetDict if multiple splits; can also be a single-split dict

    # Normalize to a DatasetDict
    if isinstance(ds, DatasetDict):
        dd = ds
    else:
        # Single split (likely 'train'); wrap to uniform handling
        dd = DatasetDict({"train": ds})

    # 2) Map to add the field
    def _add_field(example):
        example[column_name] = persona_row_to_string(example)
        return example

    updated = DatasetDict()
    for split_name, split_ds in dd.items():
        if not all_splits and split_name != "train":
            updated[split_name] = split_ds  # leave untouched
            continue
        updated[split_name] = split_ds.map(_add_field, num_proc=map_num_proc)

    # 3) Push back to the Hub
    #    Pushing a DatasetDict updates all included splits in the same repo/config.
    updated.push_to_hub(repo_id)
    return updated


# --- Example usage ---
if __name__ == "__main__":
    # Processes all splits (if multiple) and pushes back.
    add_persona_and_push(
        repo_id="thoughtworks/psychometric_personas",
        column_name="persona_string",
        map_num_proc=None,       # set to >1 for speed if large
        all_splits=True,
    )
