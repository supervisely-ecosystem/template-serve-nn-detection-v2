from typing import Dict, List, Tuple, Union
import functools
import os
import sys
import supervisely as sly
from supervisely.app.v1.app_service import AppService

# Add app root directory to system paths
app_root_directory = os.getcwd()
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
model_meta = None
local_weights_path = None
remote_weights_path = ""
if "modal.state.slyFile" in os.environ:
    remote_weights_path = os.environ['modal.state.slyFile'] 


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
    app = AppService()
    app._add_callback("get_output_classes_and_tags", get_output_classes_and_tags)
    app._add_callback("get_custom_inference_settings", get_custom_inference_settings)
    app._add_callback("get_session_info", get_session_info)
    app._add_callback("inference_image_url", inference_image_id)
    app._add_callback("inference_batch_ids", inference_batch_ids)
    app._add_callback("inference_image_url", inference_image_url)

    # Supervisely variables
    team_id = int(os.environ["context.teamId"])
    workspace_id = int(os.environ["context.workspaceId"])

    model_meta = get_classes_and_tags_fn()
    download_model()
    deploy()
    app.run()


def deploy():
    deploy_model_fn(local_weights_path)
    sly.logger.info("Model has been deployed successfully")


def inference(image_path: str) -> sly.Annotation:
    """This is a demo function to show how to inference your custom model
    on a selected image in supervsely.

    Parameters
    ----------
    state : dict
        Dict that stores application fields
    image_path : str
        Local path to image

    Returns
    -------
    sly.Annotation
        Supervisely annotation
    """

    predictions = inference_fn(image_path)
    
    image = sly.image.read(image_path)
    # This function converts model predictions into supervisely annotation format
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
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    global model_meta
    model_meta = get_classes_and_tags_fn()
    request_id = context["request_id"]
    app.send_response(request_id, data=model_meta.to_json())


@sly.timeit
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    app.send_response(request_id, data={})


@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):

    info = get_session_info_fn()

    request_id = context["request_id"]
    app.send_response(request_id, data=info)


@sly.timeit
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_url = state["image_url"]

    ext = sly.fs.get_file_ext(image_url)
    assert ext in ["png", "jpg", "jpeg"]
    local_image_path = os.path.join(app.data_dir, f"{sly.rand_str(15)}.{ext}")
    sly.fs.download(image_url, local_image_path)

    ann = inference(image_path=local_image_path)

    request_id = context["request_id"]
    app.send_response(request_id, data=ann.to_json())


@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(app.data_dir, sly.rand_str(10) + image_info.name)
    ann = inference(image_path=image_path)
    sly.fs.silent_remove(image_path)
    
    request_id = context["request_id"]
    app.send_response(request_id, data=ann.to_json())


@sly.timeit
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)
    paths = [os.path.join(app.data_dir, sly.rand_str(10) + info.name) for info in infos]
    api.image.download_paths(infos[0].dataset_id, ids, paths)
    results = []
    for image_path in paths:
        ann = inference(image_path=image_path)
        results.append(ann.to_json())

    request_id = context["request_id"]
    app.send_response(request_id, data=results)


def convert_preds_to_sly_annotation(
        predictions: dict,
        img_size: Tuple[int, int],
) -> sly.Annotation:
    """Convert model predictions to supervisely format annotation.

    Parameters
    ----------
    predictions : dict
        Prediction bounding boxes
    img_size : Tuple[int, int]
        height and width of the image

    Returns
    -------
    sly.Annotation
        Supervisely annotation in JSON format
    """
    labels = []
    for prediction in predictions:
        box = prediction["bbox"]
        sly_rect = sly.Rectangle(box[0], box[1], box[2], box[3])
        if isinstance(model_meta, sly.ProjectMeta):
            sly_obj_class = model_meta.get_obj_class(prediction["class"])
        else:
            sly_obj_class = sly.ObjClass(
                prediction["class"], sly.Rectangle
            )
        if "confidence" in prediction.keys():
            sly_tag_meta = sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
            sly_tag = sly.Tag(sly_tag_meta, prediction["confidence"])
            sly_tag_collection = sly.TagCollection([sly_tag])
        else:
            sly_tag_collection = sly.TagCollection([])
        sly_label = sly.Label(sly_rect, sly_obj_class, tags=sly_tag_collection)
        labels.append(sly_label)

    ann = sly.Annotation(img_size=img_size, labels=labels)
    return ann


def download_model() -> None:
    global local_weights_path
    if not remote_weights_path.endswith(".pth"):
        sly.logger.info("Model is not found. Template mode with example labels will be used.")
        return

    info = api.file.get_info_by_path(team_id, remote_weights_path)
    if info is None:
        raise FileNotFoundError(f"Weights file not found: {remote_weights_path}")

    sly.logger.info("Downloading model weights...")
    local_weights_path = os.path.join(app.data_dir, sly.fs.get_file_name_with_ext(remote_weights_path))
    api.file.download(
        team_id,
        remote_weights_path,
        local_weights_path,
        cache=app.cache
    )

    sly.logger.info("Model has been successfully downloaded")

