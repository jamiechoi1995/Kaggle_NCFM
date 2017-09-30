# Kaggle https://challenger.ai/competition/scene
Using Keras+TensorFlow to solve https://challenger.ai/competition/scene

This code is forked from Kaggle_NCFM and I modified it to apply for the competition above.

**Make sure that you have installed latest version of Keras since Inception_V3 is only provided 
in the latest version!**

Step1. Download dataset from https://challenger.ai/competition/scene/subject

Step2. Use ```parse_json.py``` to split the labed data into training and validation. 

Step3. Use ```train.py``` to train a Inception_V3 network. The best model and its weights will be saved as "weights.h5".

Step4. Use ```predict.py``` to predict labels for testing images and generating the submission file "submit.json".
Note that such submission results in a 0.93310 score in the leaderboard. 

----
Step5. In order to improve our ranking, we use data augmentation for testing images. The intuition behind is similar to multi-crops,
which makes use of voting ideas. ```predict_average_augmentation.py``` implements such idea and results in a 10% ranking (Public Score: 1.09) in the leaderboard.

Step 6. Note that there is still plenty of room for improvement. For example, we could split data into defferent training and valition
data by cross-validation, e.g. k-fold. Then we train k models based on these splitted data. We average the predictions output by the k models as the final submission. This strategy will result a 5% ranking (Public Score: 1.02) in the leaderboard. We will leave the implementation as a practice for readers :)

Step 7: if you wanna to improve ranking further, object detection is your next direction!

**Update and Note:** In order to use ```flow_from_directory()```, you should create a folder named **test_stg1** and put the original test_stg1 inside it.
