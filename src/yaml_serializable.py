import yaml

class YAMLSerializable():
    
    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, input_dict):
        return cls(**input_dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as fp:
            input_dict = yaml.safe_load(fp)
            return cls.from_dict(input_dict)

    def to_dict(self):
        raise NotImplementedError()

    def to_yaml(self, path):
        with open(path, 'w') as fp:
            yaml.safe_dump(self.to_dict(), fp, sort_keys=False)

    def __repr__(self):
        return yaml.dump(self.to_dict(), sort_keys=False)
