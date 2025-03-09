Github Repository link : https://api.wandb.ai/links/da24s019-indian-institute-of-technology-madras/st30khic 

Wandb Report Link : https://github.com/DA24S019/DA6401-Assignment_1.git 


The code in the .py files implement neural network and deep learning to train and test Fashion MNIST and MNIST datset.

Following is the file structure followed in the implemenetation:


1. feedforward_neural_network.py : 
In this file code is written for Class Neural Network which contains functions :
 
        train : for training the neural network
        forward_propagation :  for implementing forward 2.propagation step
        back_propagation : for implementing bsvk propagation
        evaluate : evaluation of validation accuracy and validation loss
        predict : predict the output given values of input or sample


2. activavtion_functions.py : 
In this file class of Activation Functions.

3. optimizers.py : 
In this file, code is there for implementing all the  6 optimizers.

4. dataset_loader.py : 
This python file contains code for loading dataset Fashion-MNIST or MNIST. And return training, validation and testing sets.

5. arguments.py : 
This files contain the code specifications mentioned in the assignment.

6. category_logging.py : 
This code file contains code for logging the first occurrence of each 10 in the dataset as mentioned in Question 1.

7. sweep_config.py : 
This file contains the hyperparamters for tuning the network and configured for using sweep.

8. train_sweep.py : 
Here is the code for hyperparameter tuning the network with the help of wandb sweep function.

9. best_hyperparameters.py : 
In this there is code for hyperparamters tuning through sweep and then produce confuison matrix for the test data

10. readme.md : 
This files which contains information about other files.

In the code specification  I have changed some value like beta,momentum ,beta1 and beta2 and in the class sir has stated htat these value work better.

"python train.py --wandb_entity myname --wandb_project myprojectname"
This will run the code and work as arguments specified in code_specifications.

All the sweeping and question 1 CODE and best hyperparameter coding and logging confusion matrix in wandb codes are in separate python files.



