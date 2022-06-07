import os
import sys

import supervisely as sly
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from supervisely.app.fastapi import create

import src.helpers as helpers

# Add app root directory to system paths
app_root_directory = os.getcwd()
sys.path.append(app_root_directory)
sys.path.append(os.path.join(app_root_directory, "src"))
print(f"App root directory: {app_root_directory}")
sly.logger.info(f'PYTHONPATH={os.environ.get("PYTHONPATH", "")}')

# Use the following lines only for debug purposes
debug_env_path = os.path.join(app_root_directory, "debug.env")
secret_debug_env_path = os.path.join(app_root_directory, "secret_debug.env")
load_dotenv(debug_env_path)
load_dotenv(secret_debug_env_path, override=True)

# App initialization
api = sly.Api.from_env()
app = FastAPI()
sly_app = create()
app.mount("/sly", sly_app)
# templates = sly.app.fastapi.Jinja2Templates(directory="templates")

# Global variables
team_id = int(os.environ["context.teamId"])
workspace_id = int(os.environ["context.workspaceId"])
model_classes = ["person", "car", "dog"]
model_id_classes_map = dict(enumerate(model_classes))
confidence_tag_name = "confidence"


# Read .yaml file with settings to apply it to model
settings_path = os.path.join(app_root_directory, "custom_settings.yaml")
sly.logger.info(f"Custom inference settings path: {settings_path}")
with open(settings_path, "r") as file:
    default_settings_str = file.read()
    # default_settings is also a global variable
    default_settings = yaml.safe_load(default_settings_str)

def construct_model_meta(model_classes) -> sly.ProjectMeta:
    """Generate project meta from model classes list.

    Parameters
    ----------
    model_classes : List[str]
        List of model classes

    Returns
    -------
    sly.ProjectMeta
        Supervisely project meta
    """
    colors = []
    for i in range(len(model_classes)):
        colors.append(sly.color.generate_rgb(exist_colors=colors))

    obj_classes = [
        sly.ObjClass(name, sly.Rectangle, color)
        for name, color in zip(model_classes, colors)
    ]
    tags = [sly.TagMeta(confidence_tag_name, sly.TagValueType.ANY_NUMBER)]
    meta = sly.ProjectMeta(
        obj_classes=sly.ObjClassCollection(obj_classes),
        tag_metas=sly.TagMetaCollection(tags),
    )
    return meta

meta = construct_model_meta(model_classes)
