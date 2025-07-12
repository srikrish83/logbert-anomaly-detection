import os
from dotenv import load_dotenv
import yaml

load_dotenv()

# Path to config.yaml
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "../config/config.yaml")

def load_config(path=CONFIG_FILE):
    """
    Load YAML config file into a Python dictionary.
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)

# Load once at import
config = load_config()
