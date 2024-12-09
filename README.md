# Directly connecting jupyter notebook to sagmaker 
Using AWS -Sagemaker to train Machine learing model directly from jupyter notebook

### Most of the times we use AWS UI to access , open notebook inside sagemaker , train the model and inference from endpoint but what if we configure sagemaker using CLI and access it from jupyter notebook from your local machine

>Inshort we are just using AWS UI for creating IAM role , S3 bucket and the best part is we don't have to use sagemaker UI for training and inferencing from end-point

## Steps : 

### Configure AWS CLI & IAM role

- First create IAM role from ur AWS , set poicy access as admin or sagemaker access and create access key
- Download and Install Amazon CLI and configure
> aws configure
- paste the access & secret key
-Follow the code in the notebook

### Create S3 bucket & custom script

Notebook :  AWS_Sagenaker_mL_train.ipynb [https://github.com/mahikshith/AWS_Sagemaker_ML_trainer/blob/main/AWS_Sagenaker_mL_train.ipynb]

- Create S3 bucket and upload the data to S3 using the code in the notebook
- Write a custom script [sagemaker_script_ml.py] which contains code for sagemaker to train the model
- Create an entry point so that AWS sagemaker picks the code
- Create a new role for sagemaker service- execution and copy "ARN" and assign to role variable in the notebook
- Enter all the hyper-parameters for the endpoint and RUN the job

### Model -training and inferencing : 

- When we run the job , the model training starts under training-jobs
- Now create an endpoint for inferencing
- Stop or delete the end-point once the inference is done


