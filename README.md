# AI702 G13: A Comparison of Drowsiness Detection Performance Between a Large Language Model and Different Deep Learning Systems

[dataset](https://universe.roboflow.com/yolo-yvl6h/drowsiness-fatigue_detection) | [pretrained models](https://mbzuaiac-my.sharepoint.com/personal/abdulrahman_almarzooqi_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fabdulrahman%5Falmarzooqi%5Fmbzuai%5Fac%5Fae%2FDocuments%2FBest%20Model%20Weights&ct=1714335263260&or=OWA%2DNT%2DMail&cid=22ae46c4%2D3d0a%2D2e04%2Df78a%2D5b39b386d07d&ga=1&LOF=1)

First, clone this repository and navigate to `driver_drowsiness_detection` folder:

```
git clone https://github.com/Abdulla-Almarzooqi/driver_drowsiness_detection.git
cd driver_drowsiness_detection
```

## I) YOLOv8, YOLOv9, and RT-DETR

### Setup Instructions
    
1. Create a conda environment:

     ```
     conda create -n detection
     conda activate detection
     ```

2. To run YOLOv8, we used PyTorch v1.13.1 with CUDA 11.6. For YOLOv9 and RT-DETR, we used PyTorch v2.0.0 along with CUDA 11.7. It is preferred that you use the latter only for all models. To install it, run the following command:

   ```
   conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```

3. Install Ultralytics:

   ```
   pip install ultralytics
   ```

4. Install the dataset [from this link](https://universe.roboflow.com/yolo-yvl6h/drowsiness-fatigue_detection/dataset/4/download). It is preferable that you use the download code instead of downloading a zip to your computer. Choose the format *"YOLOv8 Oriented Bounding Boxes"*, then copy the code snippet with the API key and run it.

### Demo

To run the demo for these object detection models, simply follow the instructions stated in `models_demo.ipynb` which can be found under the folder `Demo`. In this notebook, you can find a [link to our pretrained models](https://mbzuaiac-my.sharepoint.com/personal/abdulrahman_almarzooqi_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fabdulrahman%5Falmarzooqi%5Fmbzuai%5Fac%5Fae%2FDocuments%2FBest%20Model%20Weights&ct=1714335263260&or=OWA%2DNT%2DMail&cid=22ae46c4%2D3d0a%2D2e04%2Df78a%2D5b39b386d07d&ga=1&LOF=1). Download them, then place them inside the folder `models` (rest of instructions are stated in the notebook). Note that the folder `runs/detect/predict#/` is automatically created upon code execution.

### Training and Evaluation

1. **YOLOv8**: To train and evaluate the YOLOv8 model, run `drowsiness-yolov8.ipynb`. Before you run each cell, make sure you adjust the paths to match your use case (places where you need to adjust the paths are indicated by comments within the notebook). After executing this notebook, you will get a new custom YAML file to use for the training, open this file and change the class names to {0:alert, 1:drowsy} to get the class names instead of just numbers as the class name. This new file will be used for YOLOv9 and RT-DETR as well later on.

2. **YOLOv9**: To train and evaluate the YOLOv9 model, run `drowsiness-yolov9.ipynb`. Make sure to adjust the paths accordingly where stated. You will use the same YAML file created in step 1.

3. **RT-DETR**: Similarly, to train and evaluate the RT-DETR model, run `drowsiness-rtdetr.ipynb`. Make sure to adjust the paths accordingly where stated. You will also use the same YAML file created in step 1.

## II) LLaVA

### Setup Instructions

To install LLaVA, simply follow the instructions specified [in this repository](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install) (better to create a separate conda environment for this part). In case you did not install the dataset yet, refer to step 4 of section (I) in the setup instructions.

### Demo

To run the demo for LLaVA, simply follow the instructions stated in `llava_demo.ipynb` which can be found under the folder `Demo`. Note the the prompt has to be written in a specific format, which is specified within the notebook.

### Prediction and Evaluation

To use LLaVA for drowsiness classification and evaluate its performance for all of the three variants (LLaVA-7B, LLaVA-13B, and BakLLaVA), run `drowsiness-llava.ipynb`. Follow the comments within the notebook for further guidelines.

## Acknowledgements

1. For the three object detection models, we referred [to this code](https://www.kaggle.com/code/ahmedmoneimm/yolov8-drowsiness-detection). We did the following changes:
    - Tweaked the hyperparameters a bit
    - Tried different models (YOLOv9 and RT-DETR)
    - For YOLOv9 and RT-DETR, we did not use the same code to create a new custom YAML file, instead, we directly made use of the one created in the YOLOv8 notebook.

3. For LLaVA, we referred [to this notebook](https://colab.research.google.com/drive/1qsl6cd2c8gGtEW1xV5io7S8NHh-Cp1TV?usp=sharing). We tried different models and added our own code to do the following:
   - Extract the 'yes'/'no' LLM output
   - Get the predictions along with the corresponding ground truth labels
   - Compare them and get the evaluation metrics.
