This project utilizes the HC3 dataset to train and evaluate various binary classifiers making use of various model architectures and vectorisation methods.

The required libraries can be downloaded using 'pip install -r requirements.txt'

The code provides no interface and was simply ran to create each model and then create a corresponding text file containing the cross validation F1-scores, log loss and classification report

This code can be run on your machine to generate these same models referenced in the report in the final results section as well as the text files with performance metrics that are used in the data tables in the same section

The code makes use of sklearn for all model architectures and so does not require GPU's or expensive hardware although for reference the hardware of the machine the models were created on will be specified below

CPU: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
Memory: 16 GB

The dataset comes from https://huggingface.co/datasets/Hello-SimpleAI/HC3/blob/main/all.jsonl

To preprocess the data use preprocess.py with all.jsonl in the same location, you can pass the limit of the dataset you wish to use as an argument for example 1000 should limit the model to preprocess approximately 1000 sentences (sometimes more are created due to the check not happening every sentence but instead after each questions answers are fully processed)

After preprocess.py has run successfully a file called 'labeled_data.csv' will have been created and model.py can be ran to train all models consisting of all combinations of each vectorisation technique with each model architecture

Each model will create a .model file and a corresponding .txt file containing the metrics for that model
