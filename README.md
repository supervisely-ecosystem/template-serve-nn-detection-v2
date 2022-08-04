<div align="center" markdown>

<img src="https://user-images.githubusercontent.com/48913536/172667748-75e5b35e-c9e5-4c43-a9cd-a9799b5a1af8.png" style="width: 100%;"/>

# Serve Custom Detection Model Template

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#preparation">Preparation</a> •
  <a href="#how-to-develop">How To Develop</a> •
  <a href="#how-to-add-model-as-app-enterprise-edition-only">How To Add Model As App</a> •
  <a href="#how-to-run">How To Run</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/template-serve-nn-detection)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/template-serve-nn-detection)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/template-serve-nn-detection-v2.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/template-serve-nn-detection-v2.png)](https://supervise.ly)

</div>

# Overview

Template Serve NN Detection app is designed for **developers** and can be used as a starting point for creating an application for serving your own detection NN models on Supervisely.

# Preparation

**Step 1.** Make a fork from this repository

**Step 2.** Clone repository to your computer

**Step 3.** Open repo directory and create python virtual environment by running the following command from the application root directory in terminal:

```bash
python -m venv venv
```

**Step 4.** Activate virtual environment:

```bash
source venv/bin/activate
```

**Step 5.** Install requirements.txt:

```bash
pip install -r requirements.txt
```

**Note:** we provide a docker image with cuda runtime and it's dependencies, but if you need to use something specific, add it to the `requirements.txt`, or use your own docker image, please contact supervisely technical support for details

**Note 2:** you can change application name in `config.json`.

# How To Develop

**Note:** recommended Python version >= 3.8

**Details:**
By default template app generates demo predictions to demonstrate the functionality. In order to implement your custom model, you will need to edit `main.py` file only. 

`main.py` - contains 4 functions with commentaries to help you implement your custom nn model:

* `get_classes_and_tags()` - constructs ProjectMeta object with specified model classes and tags.
* `get_session_info()` - generates model info dict with any parameters (see recommended parameters in file). You will see this parameters when you will connect to your model from other apps.
* `inference(image_path)` - this functions gets input image path and return model predictions on this image. See predictions format in file. Inference results will be automatically converted to [supervisely annotation format](https://docs.supervise.ly/data-organization/00_ann_format_navi).
* `deploy_model(model_weights_path)` - function initializes model to be ready to get input data for inference.

**Step 1.** Make sure you've edited `main.py`, without edits it will generate demo predictions

**Step 2.** Run `main.py` from terminal or by using your IDE interface:

```bash
python main.py
```

**Step 3.** When your model is ready, add additional modules and packages that are required to run your served model to `requirements.txt`

**Step 4.** Add your model as private app to Supervisely Ecosystem

# How To Add Model As App [Enterprise Edition only]

**Step 1.** Go to Ecosystem page and click on `private apps`

<img src="https://user-images.githubusercontent.com/48913536/172667770-880d2c2b-2827-4fc1-ac84-1f5c6827eb66.png" style="width:80%;"/>

**Step 2.** Click `+ Add private app` button

<img src="https://user-images.githubusercontent.com/48913536/172667780-6e87d2f7-3f68-40bd-a70f-f897568f2ffb.png" style="width:80%;"/>

**Step 3.** Copy and paste repository url and generated [github/gitlab personal token](https://docs.supervise.ly/enterprise-edition/advanced-tuning/private-apps) to modal window

<img src="https://user-images.githubusercontent.com/48913536/172667782-b5678b3d-0950-4638-bd66-abae8b8a6719.png" style="width:50%;"/>

**Video**
<a data-key="sly-embeded-video-link" href="https://youtu.be/9EOFd8sjA3Q" data-video-code="9EOFd8sjA3Q">
    <img src="https://user-images.githubusercontent.com/48913536/172684900-43cf20a7-cf63-438a-b4ea-69f31f5facd5.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:20%;">
</a>

# How To Run:

**Step 1.** Upload your model to Team Files

<img src="https://user-images.githubusercontent.com/48913536/172897541-9ff48bb7-00cb-4732-9011-8604290279f7.gif" style="width:80%;"/>

**Step 2.** Add app with implemented custom nn model to your team from Ecosystem

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/template-serve-nn-detection" src="https://user-images.githubusercontent.com/48913536/172667758-d9de3332-f2ff-482f-8a34-110c1547e46e.png" width="500px" style='padding-bottom: 20px'/>  

**Step 3.** Run the application from the context menu of `.pth` file. If you are running application from file with different than `.pth` extension, app will use demo model

<img src="https://user-images.githubusercontent.com/48913536/172667785-00534a3e-ce8b-4d50-9490-c3de94afc4ac.png" style="width:80%;"/>

**Step 4.** Press the Run button in the modal window

<img src="https://user-images.githubusercontent.com/48913536/172667762-5ffbb929-d28d-4036-a52e-855fb402fdf1.png" style="width:50%;"/>

**Step 5.** Add one of the related apps to your team from Ecosystem and run it.

**Step 6.** Open running applier app and connect to app session with served model

<div>
  <table>
    <tr style="width: 100%">
      <td>
        <b>NN Image Labeling</b>
        <img src="https://user-images.githubusercontent.com/48913536/172667788-bcd75bae-079e-411f-a8f7-49bbb8e160d4.png" style="width:100%;"/>
      </td>
      <td>
        <b>Apply NN to Images Project</b>
        <img src="https://user-images.githubusercontent.com/48913536/172667799-e8151dc7-d457-4631-9231-a99f72e5bd8e.png" style="width:100%;"/>
      </td>
    </tr>
  </table>
</div>

**Step 7.** Your served model is ready to apply. 

Once you integrated serving app for your model, you can use any available inference interfaces in Ecosystem:


* [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - **Apply served model to image** 

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://user-images.githubusercontent.com/48913536/172667808-56876936-a880-4548-a216-78abaea50812.png" width="450px" style='padding-bottom: 20px'/>  

* [Apply NN to Images Project](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - **Apply served model to whole project or dataset**

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://user-images.githubusercontent.com/48913536/172667811-d5089a48-b7f2-4ddf-929c-1f4629c3fc4d.png" width="450px" style='padding-bottom: 20px'/>  

