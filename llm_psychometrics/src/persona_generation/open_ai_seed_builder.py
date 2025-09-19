import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI

from pydantic import BaseModel

class SeededExamples(BaseModel):
    seeds: List[str]

class OpenAISeedBuilder:
    """
    Populates unfilled AppearanceCategories and BehaviorCategories with 20 examples each,
    using OpenAI structured outputs (no string parsing).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 1.5,
        top_p: float = 0.98,
        dry_run: bool = False,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.dry_run = dry_run

    def load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def save_yaml(self, data: Dict[str, Any], input_path: Path) -> Path:
        out_path = input_path.parent / f"populated_{input_path.name}"
        with open(out_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"✅ Filled YAML written to {out_path}")
        return out_path

    def _few_shot_context(self, categories: Dict[str, List[str]], category_type: str) -> str:
        blocks = []
        for cat, seeds in categories.items():
            if seeds and len(seeds) >= 5:
                examples = "\n".join([f"- {s}" for s in seeds[:5]])
                blocks.append(f"{category_type}: {cat}\n{examples}")
        return "\n\n".join(blocks) if blocks else f"(No prior examples for {category_type}.)"

    def _generate_examples(
            self,
            category_name: str,
            category_type: str,
            few_shot_block: str
    ) -> List[str]:
        if self.dry_run:
            return [f"{category_name} example {i}" for i in range(1, 21)]

        prompt = (
            f"You are generating seeds for police officer personas.\n\n"
            f"Below are some example categories:\n{few_shot_block}\n\n"
            f"Now generate exactly 20 examples for {category_type}: {category_name}.\n"
            f"- Each should be 4–8 tokens long.\n"
            f"- Return as JSON list under key `seeds`."
        )

        resp = self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            text_format=SeededExamples,  # schema validation here
        )

        return resp.output_parsed.seeds

    def populate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        root = data.get("PoliceOfficerPersonaSeeds", {})
        for block in ["AppearanceCategories", "BehaviorCategories"]:
            if block in root and isinstance(root[block], dict):
                categories = root[block]
                few_shot = self._few_shot_context(categories, block)
                for cat, vals in categories.items():
                    if vals == []:
                        print(f"Populating {block} → {cat}")
                        categories[cat] = self._generate_examples(cat, block, few_shot)
        return data

    def build(self, yaml_path: str | Path) -> Path:
        yaml_path = Path(yaml_path)
        data = self.load_yaml(yaml_path)
        data = self.populate(data)
        return self.save_yaml(data, yaml_path)
