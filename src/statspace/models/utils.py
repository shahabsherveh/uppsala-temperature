import json
from typing import Any


class BaseParams:
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def parse(self) -> Any:
        return self.to_dict()

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_file(self, file_path):
        config_dict = self.parse()
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configuration saved to {file_path}")
