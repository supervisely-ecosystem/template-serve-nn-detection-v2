import functools
import os
import sys
from pathlib import Path
from typing import Tuple

import supervisely as sly
from fastapi import FastAPI
from pydantic import BaseModel
from supervisely import FileCache

# Add app root directory to system paths
app_root_directory = Path(__file__).absolute().parent
sly.logger.info(f"App root directory: {app_root_directory}")
sys.path.append(app_root_directory.as_posix())

# Use the following lines only for debug purposes
# from dotenv import load_dotenv
# debug_env_path = os.path.join(app_root_directory, "debug.env")
# secret_debug_env_path = os.path.join(app_root_directory, "secret_debug.env")
# load_dotenv(debug_env_path)
# load_dotenv(secret_debug_env_path, override=True)


class ServeRequestBody(BaseModel):
    state: dict = {}
    context: dict = {}


api: sly.Api = None
app = FastAPI()
app_temp_dir_path = app_root_directory / 'app_temp'
app_temp_dir_path.mkdir(parents=True, exist_ok=True)
team_id = None
workspace_id = None

# Template model settings
inference_fn = None
get_classes_and_tags_fn = None
get_session_info_fn = None
deploy_model_fn = None
model_meta: sly.ProjectMeta = None
local_weights_path = None
remote_weights_path = ""
if "modal.state.slyFile" in os.environ:
    remote_weights_path = os.environ['modal.state.slyFile'] 

app_cache_dir = app_temp_dir_path / 'cache'
app_cache_dir.mkdir(parents=True, exist_ok=True)
app_cache = FileCache(name="FileCache", storage_root=app_cache_dir.as_posix())


def serve_detection(get_info_fn,
                    get_meta_fn,
                    inf_fn,
                    deploy_fn):
    global api, app, team_id, workspace_id, get_session_info_fn, \
        get_classes_and_tags_fn, inference_fn, deploy_model_fn, model_meta

    get_session_info_fn = get_info_fn
    get_classes_and_tags_fn = get_meta_fn
    inference_fn = inf_fn
    deploy_model_fn = deploy_fn

    sly.logger.info("Supervisely settings", extra={
        "context.teamId": team_id,
        "context.workspaceId": workspace_id
    })

    # App initialization
    api = sly.Api.from_env()

    app.add_api_route("get_output_classes_and_tags", get_output_classes_and_tags, methods=["POST"])
    app.add_api_route("get_custom_inference_settings", get_custom_inference_settings, methods=["POST"])
    app.add_api_route("get_session_info", get_session_info, methods=["POST"])
    app.add_api_route("inference_image_url", inference_image_id, methods=["POST"])
    app.add_api_route("inference_batch_ids", inference_batch_ids, methods=["POST"])
    app.add_api_route("inference_image_url", inference_image_url, methods=["POST"])

    # Supervisely variables
    team_id = int(os.environ["context.teamId"])
    workspace_id = int(os.environ["context.workspaceId"])

    model_meta = get_classes_and_tags_fn()
    download_model()
    deploy()


def deploy():
    deploy_model_fn(local_weights_path)
    sly.logger.info("Model has been deployed successfully")


def inference(image_path: str) -> sly.Annotation:
    # User custom inference
    predictions = inference_fn(image_path)
    
    image = sly.image.read(image_path)
    
    # This function converts custom model predictions into supervisely annotation format
    annotation = convert_preds_to_sly_annotation(
        predictions,
        img_size=image.shape[:2],
    )

    return annotation


def draw_demo_result(predictions, input_image_path, output_image_path) -> None:
    image = sly.image.read(path=input_image_path)
    annotation = convert_preds_to_sly_annotation(predictions, image.shape[:2])
    for label in annotation.labels:
        label.draw_contour(image, thickness=5)
    
    sly.image.write(output_image_path, image)
    sly.logger.info(f"Labeled image saved to {output_image_path} successfully")


def convert_preds_to_sly_annotation(
        predictions: dict,
        img_size: Tuple[int, int],
) -> sly.Annotation:
    """Convert model predictions to supervisely format annotation."""
    labels = []
    for prediction in predictions:

        if "bbox" not in prediction.keys():
            raise ValueError(f"Model prediction does not contain bounding box. Prediction dict: {prediction}")
        if "class" not in prediction.keys():
            raise ValueError(f"Prediction does not contain class name. Prediction dict: {prediction}")
        box = prediction["bbox"]
        sly_rect = sly.Rectangle(box[0], box[1], box[2], box[3])
        if model_meta is not None:
            try:
                sly_obj_class = model_meta.get_obj_class(prediction["class"])
            except KeyError as e:
                raise ValueError("Predicted class name was not added to \
                    get_classes_and_tags(). Please add this class and try again.")
        else:
            sly_obj_class = sly.ObjClass(prediction["class"], sly.Rectangle)
             
        if "confidence" in prediction.keys():
            if model_meta is not None:
                try:
                    sly_tag_meta = model_meta.get_tag_meta("confidence")
                except KeyError as e:
                    raise ValueError("Predicted confidence tag name was not added to \
                    get_classes_and_tags(). Please add this tag and try again.")
            else:
                sly_tag_meta = sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
                
            if not isinstance(prediction["confidence"], float):
                raise TypeError(f"Predicted confidence of type {type(prediction['confidence'])} \
                    must be float.")
            elif prediction["confidence"] < 0 or prediction["confidence"] > 1:
                raise ValueError(f"Predicted confidence value {prediction['confidence']} \
                    must be in range [0, 1].")
            else:
                sly_tag = sly.Tag(sly_tag_meta, prediction["confidence"])
            
            sly_tag_collection = sly.TagCollection([sly_tag])
        else:
            sly_tag_collection = sly.TagCollection([])
        sly_label = sly.Label(sly_rect, sly_obj_class, tags=sly_tag_collection)
        labels.append(sly_label)

    ann = sly.Annotation(img_size=img_size, labels=labels)
    return ann


def download_model() -> None:
    """Download model weights from Supervisely team files."""
    if not remote_weights_path.endswith(".pth"):
        sly.logger.info("Model is not found. Template mode with example labels will be used.")
        return

    global local_weights_path
    info = api.file.get_info_by_path(team_id, remote_weights_path)
    if info is None:
        raise FileNotFoundError(f"Weights file not found: {remote_weights_path}")

    sly.logger.info("Downloading model weights...")
    local_weights_path = os.path.join(app_temp_dir_path, sly.fs.get_file_name_with_ext(remote_weights_path))
    api.file.download(
        team_id,
        remote_weights_path,
        local_weights_path,
        cache=app_cache
    )

    sly.logger.info("Model has been successfully downloaded")


#################################
# REQUESTS PROCESSING FUNCTIONS #
#################################

def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@sly.timeit
@send_error_data
def get_output_classes_and_tags():
    global model_meta
    model_meta = get_classes_and_tags_fn()
    return model_meta.to_json()


@sly.timeit
@send_error_data
def get_custom_inference_settings():
    return {}


@sly.timeit
@send_error_data
def get_session_info():
    return get_session_info_fn()


@sly.timeit
@send_error_data
def inference_image_url(request_body: ServeRequestBody):
    state = request_body.state
    sly.logger.debug("Input data", extra={"state": state})
    image_url = state["image_url"]

    ext = sly.fs.get_file_ext(image_url)
    assert ext in ["png", "jpg", "jpeg"]

    local_image_path = os.path.join(app_temp_dir_path, f"{sly.rand_str(15)}.{ext}")
    sly.fs.download(image_url, local_image_path)

    ann = inference(image_path=local_image_path)

    return ann.to_json()


@sly.timeit
@send_error_data
def inference_image_id(request_body: ServeRequestBody):
    state = request_body.state

    sly.logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(app_temp_dir_path, sly.rand_str(10) + image_info.name)
    ann = inference(image_path=image_path)
    sly.fs.silent_remove(image_path)

    return ann.to_json()


@sly.timeit
@send_error_data
def inference_batch_ids(request_body: ServeRequestBody):
    state = request_body.state

    sly.logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)
    paths = [os.path.join(app_temp_dir_path, sly.rand_str(10) + info.name) for info in infos]
    api.image.download_paths(infos[0].dataset_id, ids, paths)
    results = []
    for image_path in paths:
        ann = inference(image_path=image_path)
        results.append(ann.to_json())

    return results
