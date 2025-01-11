import yaml


class ContextFreeGrammar:
    def __init__(self):
        self.cfg_rules = {}
        self.start_symbol = None

    def load_cfg(self, file_path):
        """
        Loads the CFG from a YAML file.
        """
        # Detect file extension and load accordingly
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            with open(file_path, 'r') as file:
                cfg_data = yaml.safe_load(file)
        else:
            raise ValueError("Unsupported file format. Use .yaml or .yml")

        # Set start symbol and cfg rules
        self.start_symbol = cfg_data.pop('start_symbol', None)  # Load start symbol
        self.cfg_rules = cfg_data  # Assign the remaining data to cfg_rules
