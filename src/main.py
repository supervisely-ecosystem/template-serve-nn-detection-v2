import supervisely as sly
import helpers
import sly_globals as g


def get_classes_and_tags():
    classes = [
        # Example
        sly.ObjClass("person", sly.Rectangle),
        sly.ObjClass("car", sly.Rectangle),
        sly.ObjClass("bus", sly.Rectangle)

        # Put any needed classes here ....
    ]
    
    classes = sly.ObjClassCollection(classes)

    tags = [
        # Example: confidence tag for bounding boxes
        sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
    ]
    tags = sly.TagMetaCollection(tags)
    return sly.ProjectMeta(obj_classes=classes, tag_metas=tags)
    
    
def get_session_info():
    return {
        # Recommended info values
        "app": "Serve Custom Detection Model Template",
        "model_name": "Put your model name",
        "device": "cpu", 
        "classes_count": 3,
        "tags_count": 1,
        "sliding_window_support": False
        # Put any key-value that you want to ....
    }


def inference(image_path):
    image = sly.image.read(path=image_path) # shape: [H, W, 3], RGB
    
    #########################
    # INSERT YOUR CODE HERE #
    #########################
    # predictions = model_inference(image)

    # example (remove it when you'll use your own predictions)
    predictions = [
        {
            "bbox": [50, 100, 77, 145], # [top, left, bottom, right]
            "class": "person", # class name like in get_classes_and_tags() function
            "confidence": 0.88 # optional 
        }
    ]
    return predictions
    

def deploy_model(model_weights_path):
    #########################
    # INSERT YOUR CODE HERE #
    #########################
    # example:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = init_model(weigths=model_weights_path, device=device)
    
    model = None
    return model
    
    
def main():
    helpers.serve_detection(
        get_session_info, 
        get_classes_and_tags, 
        inference, 
        deploy_model
    )


if __name__ == "__main__":
    sly.main_wrapper("main", main)
