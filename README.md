# tf_deep_vision_public

# User Manual :

## Creating a dataset

A dataset is in the form of a json file. Here is an example of a dataset file : 


```json
{
	"dataset_name": "dataset_name"
	"input_image_size": 32, 
	"output_image_size": 16,
	"labels_map": {"Robot": 4, "Empty": 1, "Ball": 0, "T": 5, "PostBase": 3, "X": 6, "LineCorner": 2},
	"inverted_labels_map": {"0": "Ball", "1": "Empty", "2": "LineCorner", "3": "PostBase", "4": "Robot", "5": "T", "6": "X"},
	"features_used": ["Ball", "Empty", "PostBase", "Robot", "LineCorner", "T", "X"],
	"train_proportion": 0.9,
	"training_set": [...],
	"validation_set": [...],
}
```

The `training_set` and `validation_set` fields contain dictionaries of the form `{image_path : one_hot_encoded_label}` .

You can use `dataset.py` to generate such file. In the main function, you can set the following parameters of the dataset : 

* dataset_name : quite straightforward
* data_path : a list of of directories containing the data. The data must be formatted as following :
  * A directory contains images and their associated json metadata. 
  * The png image and the json file must have the same name.
  * The json file contains information on the features present in the image. 
  * It can contain any number of features, and if there are none, just write `null` at the beggining of the file.
  * Here is an example of such a file : 
```json
{
   "Ball" : [
      {
         "center" : [ 19.487667083740234, 13.138755798339844 ],
         "distance" : 2.0170728902695156
      }
   ],
   "PenaltyMark" : [
      {
         "center" : [ 26.975473403930664, 21.185789108276367 ],
         "distance" : 2.0523011199808363
      }
   ]
}
```


* features_used : unordered list of features that will be used in this dataset. Be careful to use the same names as in the json metadata
* input_image_size : the size of the images of the dataset
* output_image_size : if you want the images to be rescaled, put a different value than `input_image_size`
* train_proportion : the proportion of data that will be put in the training set. A value of `0.9` means that 90% of the data will be in the training set, and 10% in the validation set

Then, just execute : 

	$ python dataset.py

A json file with the dataset_name you chose will be created in the current directory, as well as a `VISU` directory that allows you to check the dataset content. A subdirectory is created for each feature, and features combinations (images that contain multiple features).


## Training

Once your dataset is created, you can start training.

In `train.py`, you can edit the following hyperparameters : 

* batch_size
* nb_epochs
* learning_rate
* model_params : Here you can set the hyperparameters of the network 

Then, you can execute : 

	$ python train.py -g <gpu_id> -ds <dataset.json>
	
For a list of arguments, execute `train.py` with `-h`.

You will be prompted for the name of your experiment and a comment about it. Just hit return to leave to default.

During training, you can monitor the learning with tensorboard. In another terminal, `cd` to the directory of the currently training model and execute : 

	$ tensorboard -logdir="logs/" -port <port_number>

Then, just open the provided url in an internet browser.

At each epoch, a `model.pb` file is generated. You can then use it in your application. For example, it is quite easy to use `OpenCV` to perform inference on a .pb model using the `dnn` module.


## Tools 

* `preprocessImages.py` : you can use this script to check if there are bad images in your data. Set `dataPath` as in `dataset.py` and execute without arguments
* `infer.py` : you can use this script to perform inference on a directory containing images. See `$ python infer.py -h` for a list of arguments.
