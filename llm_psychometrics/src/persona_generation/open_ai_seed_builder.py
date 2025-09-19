import yaml
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI


class OpenAISeedBuilder:
    """
    Populates unfilled AppearanceCategories and BehaviorCategories with 20 examples each,
    using OpenAI with few-shot examples from already-populated categories.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def save_yaml(self, data: Dict[str, Any], input_path: Path):
        out_path = input_path.parent / f"populated_{input_path.name}"
        with open(out_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"✅ Filled YAML written to {out_path}")
        return out_path

    def _few_shot_context(self, categories: Dict[str, List[str]], category_type: str) -> str:
        """
        Build a few-shot block from already populated categories of the same type.
        """
        shots = []
        for cat, seeds in categories.items():
            if seeds and len(seeds) >= 5:  # only use non-empty
                examples = "\n".join([f"- {s}" for s in seeds[:5]])
                shots.append(f"{category_type}: {cat}\n{examples}")
        return "\n\n".join(shots)

    def _generate_examples(
        self,
        category_name: str,
        category_type: str,
        few_shot_block: str
    ) -> List[str]:
        """
        Ask OpenAI for 20 examples for the given category with few-shot context.
        """
        prompt = (
            f"You are helping build seeds for police officer personas.\n"
            f"Below are some example categories with example entries.\n\n"
            f"{few_shot_block}\n\n"
            f"Now generate 20 short, realistic, concrete examples "
            f"for {category_type}: {category_name}.\n"
            f"- Each example should be between 4 and 8 tokens.\n"
            f"- Format the output as a plain numbered list.\n"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.5,
            top_p=0.98,
        )
        content = resp.choices[0].message.content
        # Parse numbered list into clean list of strings
        examples = [
            line.split(".", 1)[-1].strip()
            for line in content.splitlines()
            if line.strip()
        ]
        return examples[:20]

    def populate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in empty appearance/behavior categories.
        """
        seeds = data.get("PoliceOfficerPersonaSeeds", {})

        for category_type in ["AppearanceCategories", "BehaviorCategories"]:
            if category_type in seeds:
                categories = seeds[category_type]
                # Build few-shot block from filled categories
                few_shot_block = self._few_shot_context(categories, category_type)
                for cat, vals in categories.items():
                    if vals == []:  # empty list
                        print(f"Populating {category_type} → {cat}")
                        categories[cat] = self._generate_examples(cat, category_type, few_shot_block)
        return data

    def build(self, yaml_path: str | Path) -> Path:
        """
        Read a YAML file, populate missing seeds, and write to
        'populated_<filename>.yaml' in the same directory.
        """
        yaml_path = Path(yaml_path)
        data = self.load_yaml(yaml_path)
        data = self.populate(data)
        return self.save_yaml(data, yaml_path)
