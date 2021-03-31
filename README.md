#### Project 4
##### author: code monkas


Information about the programs in part A:
It has several programs: main.py, rnn.py, partA.py, NBClassiffier.ipynb;

First put train.csv, dev.csv, test.csv in the same directory

The programs will produce text files called "Prediction.csv" as the program output

##### Running the program:
---------------------
To run RNN model: 
        
	$ python main.py  
	
To run word embedding model: 

	$ python partA.py
    
      or, run in the juptyer notebook with the file story_vec_similarity.ipynb
    
To run NBclassifier model: 
    
      run in the juptyer notebook with the file NBClassiffier.ipynb
      

Information about the programs in part B:
It has several programs: run_multiple_choice.py, utils_multiple_choice.py;

As for setup, please read instructions in https://github.com/huggingface/transformers and https://github.com/huggingface/transformers/tree/master/examples#multiple-choice

After that, please replace the utils_multiple_choice.py of the original one with the new one. Replace the train.csv with new one. Replace the val.csv with dev.csv, and rename the file name as val.csv. Rename the last column of the dev.csv as ‘label’. Run as instructed in the terminal, so that you can get the development accuracy as well as the model trained by the training set.

In order to get test result, please replace run_multiple_choice.py with a new one. Run as instructed, data in the output file is the prediction result. The programs will produce text files called "Prediction.csv" as the program output.


    



