Edit Resource: 
# Train TAO Action Recognition using Quick Deploy

This is part 1 of TAO workflow on Vertex AI. For part 2, refer to [Inference TAO Action Recognition using Quick Deploy](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deploy_action_tao) resource.

## NVIDIA TAO Toolkit on Google Vertex AI
The [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit), an AI training toolkit which simplifies the model training and inference optimization process using pretrained models and simple CLI interface. The result is an ultra-streamlined workflow. Bring your own models or use NVIDIA pre-trained models and adapt them to your own real or synthetic data, then optimize for inference throughput. All without needing AI expertise or large training datasets.

TAO Toolkit workflows can be deployed on Google Vertex AI using the Quick Deploy. 

## About Quick Deploy
The quick deploy feature automatically sets up the Vertex AI instance with an optimal configuration, preloads the dependencies, runs the software from NGC without any need to set up the infrastructure.

## TAO Action Recognition
In this workflow, you will train and optimize an action recognition model using [ActionRecognitionNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet) pretrained model and TAO. TAO Action Recognition is a configurable model to train a 2D or 3D neural network using the ResNet backbone. The pretrained model that you use for training has been trained on 5 classes from the [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) dataset. More information about this model can be found in [ActionRecognitionNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet) model card. 

## Get Started with TAO:
To help you get started, we have created a few Jupyter Notebooks that can be easily deployed on Vertex AI using NGC’s quick deploy feature. This feature automatically sets up the Vertex AI instance with an optimal configuration needed for training the model.

The workflow is divided into 2 Jupyter notebooks - one for training and one for model inference and optimization. 

### Model Training
Use the notebook in this resource for model training. In this notebook, you will train a 3D action recognition model using HMDB51 dataset. In this notebook, you will only train on a handful of actions but you can modify the ‘spec’ files to add more actions. 

#### Learning Objective
In this notebook, you will learn how to leverage the simplicity and convenience of TAO to:
* Train 3D RGB only for action recognition on the subset of [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)  dataset.
* Evaluate the trained model.

Simply click on the button that reads “**Deploy to Vertex AI**” and follow the instructions.

*Note*: A customized kernel for the Jupyter Notebook is used as the primary mechanism for deployment. This kernel has been built on the TAO Toolkit container. For more information on the container itself, please refer to this link for more information:

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/containers/tao-toolkit

The container version for this notebooks is **nvcr.io/nvidia/tao/tao-toolkit:4.0.0-pyt**

### Model Optimization and Inference
To evaluate and run inference on the trained model please refer to the [Inference TAO Action Recognition using Quick Deploy](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deploy_action_tao) resource.

## License <a class="anchor" name="license"></a>

By pulling and using the TAO Toolkit container, you accept the terms and conditions of these [licenses](https://developer.nvidia.com/tao-toolkit-software-license-agreement).

## Technical blogs <a class="anchor" name="technical_blogs"></a>

- [Train like a ‘pro’ without being an AI expert using TAO AutoML](https://developer.nvidia.com/blog/training-like-an-ai-pro-using-tao-automl/)
- [Developing and Deploying AI-powered Robots with NVIDIA Isaac Sim and NVIDIA TAO](https://developer.nvidia.com/blog/developing-and-deploying-ai-powered-robots-with-nvidia-isaac-sim-and-nvidia-tao/)
- Learn endless ways to adapt and supercharge your AI workflows with TAO - [Whitepaper](https://developer.nvidia.com/tao-toolkit-usecases-whitepaper/1-introduction)
- [Customize Action Recognition with TAO and deploy with DeepStream](https://developer.nvidia.com/blog/developing-and-deploying-your-custom-action-recognition-application-without-any-ai-expertise-using-tao-and-deepstream/)
- Read the 2 part blog on training and optimizing 2D body pose estimation model with TAO - [Part 1](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-1)  |  [Part 2](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-2)
- Learn how to train [real-time License plate detection and recognition app](https://developer.nvidia.com/blog/creating-a-real-time-license-plate-detection-and-recognition-app) with TAO and DeepStream.
- Model accuracy is extremely important, learn how you can achieve [state of the art accuracy for classification and object detection models](https://developer.nvidia.com/blog/preparing-state-of-the-art-models-for-classification-and-object-detection-with-tao-toolkit/) using TAO

## Suggested reading <a class="anchor" name="suggested_reading"></a>

- More information on about TAO Toolkit and pre-trained models can be found at the [NVIDIA Developer Zone](https://developer.nvidia.com/tao-toolkit)
- [TAO documentation](https://docs.nvidia.com/tao/tao-toolkit/index.html)
- Read the [TAO getting Started](https://docs.nvidia.com/tao/tao-toolkit/text/tao_quick_start_guide.html) guide and [release notes](https://docs.nvidia.com/tao/tao-toolkit/text/release_notes.html).
- If you have any questions or feedback, please refer to the discussions on [TAO Toolkit Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/tao-toolkit/17)
- Deploy your models for video analytics application using DeepStream. Learn more about [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)

## Ethical AI <a class="anchor" name="ethical_ai"></a>

NVIDIA’s platforms and application frameworks enable developers to build a wide array of AI applications. Consider potential algorithmic bias when choosing or creating the models being deployed. Work with the model’s developer to ensure that it meets the requirements for the relevant industry and use case; that the necessary instruction and documentation are provided to understand error rates, confidence intervals, and results; and that the model is being used under the conditions and in the manner intended.

***************************************


Edit Resource: 
# Inference TAO Action Recognition using Quick Deploy

This is part 2 of TAO workflow on Vertex AI. For part 1, refer to [Train TAO Action Recognition using Quick Deploy](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/tao_action_train) resource.

## NVIDIA TAO Toolkit on Google Vertex AI
The [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit), an AI training toolkit which simplifies the model training and inference optimization process using pretrained models and simple CLI interface. The result is an ultra-streamlined workflow. Bring your own models or use NVIDIA pre-trained models and adapt them to your own real or synthetic data, then optimize for inference throughput. All without needing AI expertise or large training datasets.

TAO Toolkit workflows can be deployed on Google Vertex AI using the Quick Deploy. 

## About Quick Deploy
The quick deploy feature automatically sets up the Vertex AI instance with an optimal configuration, preloads the dependencies, runs the software from NGC without any need to set up the infrastructure.

## TAO Action Recognition
In this workflow, you will optimize and run inference on an action recognition model using [ActionRecognitionNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet) pretrained model and TAO. TAO Action Recognition is a configurable model to train a 2D or 3D neural network using the ResNet backbone. The pretrained model that you use for training has been trained on 5 classes from the [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) dataset. More information about this model can be found in [ActionRecognitionNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet) model card. 

## Get Started with TAO:
To help you get started, we have created a few Jupyter Notebooks that can be easily deployed on Vertex AI using NGC’s quick deploy feature. This feature automatically sets up the Vertex AI instance with an optimal configuration needed for training the model.

The workflow is divided into 2 Jupyter notebooks - one for training and one for model model inference and optimization. 

### Model Optimization and Inference
Use the notebook in this resource for model inference and optimization. In this notebook, you will run inference on a 3D action recognition model. 

#### Learning Objective
In this notebook, you will learn how to leverage the simplicity and convenience of TAO to:
* Use a Trained 3D RGB model for action recognition on the subset of HMDB51 dataset.
* Evaluate the trained model.
* Run Inference on the trained model.
* Export the trained model to a .etlt file for deployment to DeepStream.



Simply click on the button that reads “**Deploy to Vertex AI**” and follow the instructions.

*Note*: A customized kernel for the Jupyter Notebook is used as the primary mechanism for deployment. This kernel has been built on the TAO Toolkit container. For more information on the container itself, please refer to this link for more information:

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/containers/tao-toolkit

The container version for this notebooks is **nvcr.io/nvidia/tao/tao-toolkit:4.0.0-deploy**

### Model Training
To train and fine-tune Action Recognition on your dataset, please refer to the [Train TAO Action Recognition using Quick Deploy](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/tao_action_train) resource.

## License <a class="anchor" name="license"></a>

By pulling and using the TAO Toolkit container, you accept the terms and conditions of these [licenses](https://developer.nvidia.com/tao-toolkit-software-license-agreement).

## Technical blogs <a class="anchor" name="technical_blogs"></a>

- [Train like a ‘pro’ without being an AI expert using TAO AutoML](https://developer.nvidia.com/blog/training-like-an-ai-pro-using-tao-automl/)
- [Developing and Deploying AI-powered Robots with NVIDIA Isaac Sim and NVIDIA TAO](https://developer.nvidia.com/blog/developing-and-deploying-ai-powered-robots-with-nvidia-isaac-sim-and-nvidia-tao/)
- Learn endless ways to adapt and supercharge your AI workflows with TAO - [Whitepaper](https://developer.nvidia.com/tao-toolkit-usecases-whitepaper/1-introduction)
- [Customize Action Recognition with TAO and deploy with DeepStream](https://developer.nvidia.com/blog/developing-and-deploying-your-custom-action-recognition-application-without-any-ai-expertise-using-tao-and-deepstream/)
- Read the 2 part blog on training and optimizing 2D body pose estimation model with TAO - [Part 1](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-1)  |  [Part 2](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-2)
- Learn how to train [real-time License plate detection and recognition app](https://developer.nvidia.com/blog/creating-a-real-time-license-plate-detection-and-recognition-app) with TAO and DeepStream.
- Model accuracy is extremely important, learn how you can achieve [state of the art accuracy for classification and object detection models](https://developer.nvidia.com/blog/preparing-state-of-the-art-models-for-classification-and-object-detection-with-tao-toolkit/) using TAO

## Suggested reading <a class="anchor" name="suggested_reading"></a>

- More information on about TAO Toolkit and pre-trained models can be found at the [NVIDIA Developer Zone](https://developer.nvidia.com/tao-toolkit)
- [TAO documentation](https://docs.nvidia.com/tao/tao-toolkit/index.html)
- Read the [TAO getting Started](https://docs.nvidia.com/tao/tao-toolkit/text/tao_quick_start_guide.html) guide and [release notes](https://docs.nvidia.com/tao/tao-toolkit/text/release_notes.html).
- If you have any questions or feedback, please refer to the discussions on [TAO Toolkit Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/tao-toolkit/17)
- Deploy your models for video analytics application using DeepStream. Learn more about [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)

## Ethical AI <a class="anchor" name="ethical_ai"></a>

NVIDIA’s platforms and application frameworks enable developers to build a wide array of AI applications. Consider potential algorithmic bias when choosing or creating the models being deployed. Work with the model’s developer to ensure that it meets the requirements for the relevant industry and use case; that the necessary instruction and documentation are provided to understand error rates, confidence intervals, and results; and that the model is being used under the conditions and in the manner intended.


