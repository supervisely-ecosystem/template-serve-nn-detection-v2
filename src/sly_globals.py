import os
import sys
import supervisely as sly
import pathlib

# Add app root directory to system paths
app_root_directory = str(pathlib.Path(sys.argv[0]).parents[1])
sly.logger.info(f"App root directory: {app_root_directory}")
sys.path.append(app_root_directory)

# Use the following lines only for debug purposes
# from dotenv import load_dotenv
# debug_env_path = os.path.join(app_root_directory, "debug.env")
# secret_debug_env_path = os.path.join(app_root_directory, "secret_debug.env")
# load_dotenv(debug_env_path)
# load_dotenv(secret_debug_env_path, override=True)

api = None
app = None
team_id = None
workspace_id = None

# Template model settings
inference_fn = None
get_classes_and_tags_fn = None
get_session_info_fn = None
deploy_model_fn = None
device = None
model_meta = None
local_weights_path = None
remote_weights_path = ""
if "modal.state.slyFile" in os.environ:
    remote_weights_path = os.environ['modal.state.slyFile'] 

