import os
import sys
import supervisely as sly
import yaml
from supervisely.app.v1.app_service import AppService

# Add app root directory to system paths
app_root_directory = os.path.dirname(os.getcwd())
sys.path.append(app_root_directory)
print(f"App root directory: {app_root_directory}")
sly.logger.info(f'PYTHONPATH={os.environ.get("PYTHONPATH", "")}')

# Use the following lines only for debug purposes
from dotenv import load_dotenv
debug_env_path = os.path.join(app_root_directory, "debug.env")
secret_debug_env_path = os.path.join(app_root_directory, "secret_debug.env")
load_dotenv(debug_env_path)
load_dotenv(secret_debug_env_path, override=True)

# App initialization
api = sly.Api.from_env()
app = AppService()

# Supervisely variables
team_id = int(os.environ["context.teamId"])
workspace_id = int(os.environ["context.workspaceId"])

# Template model settings
model_classes = ["person", "car", "dog"]  # this is an example list of custom nn model classes
inference_fn = None
model_id_classes_map = dict(enumerate(model_classes))
confidence_tag_name = "confidence"
model_meta: sly.ProjectMeta = None
model_name = "Random boxes"
device = None
remote_weights_path = None
local_weights_path = None
# Use this for using model from team files
# remote_weights_path = os.environ['modal.state.slyFile'] 

# Read custom_settings.yaml file
settings_path = os.path.join(app_root_directory, "custom_settings.yaml")
sly.logger.info(f"Custom inference settings path: {settings_path}")
with open(settings_path, "r") as file:
    default_settings_str = file.read()
    default_settings = yaml.safe_load(default_settings_str)
