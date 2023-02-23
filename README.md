# Fulhaus Assessment Repository. Candidate: Ashish Bhyravabhottla
===========================================================================

Github repository: [Github](https://github.com/ashish-AIML/fulhaus_assessment)

## Machine Learning Pipeline ##

This assessment is about creating a complete machine learning pipeline. It contains 4 steps:
1. Training a deep learning model for image classification
2. Building an API to access the trained model
3. Creating a Docker for accessing the API. This step is crucial for deployment
4. Implementing CI/CD actions

--------------------------------------------------------------------------
Summary of Code files:
1. Model training:
[model_train.ipynb](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/model_train.ipynb) is for training the model using ResNet50 architecture.\

2. Flask API and Postman:
[app.py](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/app.py) is the python code for building Flask and Postman API.\

3. Docker:
[Docker](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/Dockerfile) is Dockerfile. It creates container and image.\
[requirements.txt](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/requirements.txt) is for installing the pre-requisite requirements.\

4. CI/CD:
[MalDetection.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/master/MalDetection.ipynb) is to run the machine learning models and ensemble models.
--------------------------------------------------------------------------


# Step 1: Model Training #

The image classification model is designed using `tensorflow` package. The architecture used is `ResNet50` and trained with pre-trained `ImageNet` weights. 

## Dataset ##
Given dataset contains `3 classes` namely `sofa`, `chair`, `bed`. Total images were `360`, where each class is having `120` images.I have divided the dataset into `train` and `test` folders. I have kept test dataset as `20%`, i.e., trainset of total 300 images, 100 for each class and testset of 60 images, 20 for each class. 

## Understanding the model training code ##

Change the dataset directory path in the following variable: 
```python
data_dir = '/content/drive/MyDrive/fulhaus/Dataset_refined'
```

Define the number of classes in this variable:
```python
num_classes = 3
```

Define batch size and number of epochs in these following variables:
```python
batch_size = 32
epochs = 10
```

Now, I am defining the train and test paths using `os.path.join` function:
```python
# Generate training data batches
train_data = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Generate validation data batches
valid_data = valid_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
```

Now, I am not training the model from the scratch, hence, I am are using `ImageNet` weights and training the output layers:
```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_size+(3,))
```

After training the model, save the trained model using `model.save` command:
```python
model.save('/content/drive/MyDrive/fulhaus/resnet.h5')
```

After training the accuracy results are:
![Training Accuacy](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/training.png)

To execute the code, you can run each cell in the `Google Colab` by saving the runtime as `GPU`. I have used the Google Colab (GPU) to train the model. 

# Step 2: Flask API and Postman Integration #

The code for flask api is [app.py](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/app.py). 

## Understanding the API code ##

Make sure the path of the trained model is correct: 
```python
model_path = "C:\uottawa\other_stuff\job_apps\fulhaus\resnet.h5"
```
Make sure to have these in the code section:
```python
@app.route('/predict', methods=['GET'])
```

Change your class names:
```python
class_names = ['bed', 'chair', 'sofa'] 
```

That's it! now its's time to run the code. The code can is run by `python app.py` in the command prompt. After running the code, we will get the IP address where it's connected.

Now, open the Postman app, go to the workspace section and change the http section to `GET` and enter the IP address followed by `/predict`. 

Example:`http://localhost:5000/predict`

Now, go to the `Body` section in the postman, and enter `image` in the `key` section and choose `file` option in the key and select the image file in the `value` section. Now, `send` the request and the result will be in `json` format such as:
![result](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/json_results.png)

# Step 3: Docker Image & Containers #

Now, after API is successfully running, now it's time to design everything with Docker file. There are docker image and containers created after running the `Dockerfile` script. This is the [Dockerfile](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/Dockerfile), it should be simply saved as `Dockerfile` with capital `D` and without any extension. 

Open `Docker` app in the local system. Now, Dockerfile script needs to be created. In the Dockerfile script, the working directory is initiated as `/app`. The command `COPY`, copies the requirements.txt, then runs the requirements.txt, then copies all files into the `/app`. Now, the port number is initiated with `EXPOSE 5000` and finally the run commands: `CMD ["python3", "app.py"]` are declared. 

Now, we should create [docker-compose.yml](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/docker-compose.yml) file.

After creating the docker files, run the following commands in the command prompt:

1. Build. This creates image. It creates the images according to the build instructions specified in each service's `Dockerfile`. Using this command will rebuild the images to reflect any changes made to the Dockerfiles or dependencies listed in the `requirements.txt` file or in the app.py or any other file which is related to the Dockerfile. 

Type this command in the command prompt:
```python
docker-compose build
```

2. Up. All of the services listed in the `docker-compose.yml` file are started using the `docker-compose up` command. For all the services in the configuration, it generates and launches containers. If the service images are missing, docker-compose up creates them. Also, it attaches the container output to the console so that you can monitor the live logs of all the services that are now operating.

Type this command in the command prompt:
```python
docker-compose up
```

Now, after executing the above 2 commands, go to the Postman, give the IP address, select the test image and send the request. You will get the result as shown here:
![Docker Test image 1](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/docker_test1.png)
![Docker Test image 2](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/docker_test2.png)

The command prompt after executing docker looks like:
![Command Prompt](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/docker_cmd_output.png)

Images and containers in the Docker app:
![Docker Image](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/docker_image.png)

# Step 4: CI/CD Github Actions #

So, in the above steps, take for example `step 3`, if we change anything in any of the code/script, we have to run `docker-compose build` and `docker-compose up` everytime. Instead of that, we use CI/CD method to automatically update. So, for setting up the CI/CD using Github actions, we go to out repository, then, we go to `Action`, select `New Workflow`. Inside this, we then select `Docker Image` and configured it. 

Hence, I have created this following script: [docker-image.yml](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/.github/workflows/docker-image.yml)

Now, if any of the code is changed, the `docker-image.yml` script is executed. We can go the `Actions` workflow and check the status as shown here:
![Github Actions](https://github.com/ashish-AIML/fulhaus_assessment/blob/main/actions.png)


--------------------------------------------------------------------------

So, we have:
1. Trained an image classification model, saved the weights
2. Build an API using Flask and visualised the requests and outputs in the Postman
3. Created Dockerfile of the API
4. Created Github Actions for CI/CD in the Github