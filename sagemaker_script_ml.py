
# random forest classifier 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import numpy as np 
import argparse
import os 
import joblib
import boto3 
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# custome script : 

def function_for_model(model_dr): 
  classifier = joblib.load(os.path.join(model_dr,"model.joblib"))
  # loads the model from joblib model 

if __name__ == "__main__":

  # Arguements needs for sklearn-sagemaker 
  # Hyper-parmeters
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_estimators",type=int,default=110)
  parser.add_argument("--random_state",type=int,default=42)
  parser.add_argument("--max_depth",type=int,default=15)
  parser.add_argument("--n_jobs",type=int,default=-1)

  # data , model , output needs to stored in sagemaker environment 

  parser.add_argument("--model-dir",type=str,default=os.environ.get("SM_model_trained_DIR"))
  parser.add_argument("--train",type=str,default=os.environ.get("SM_train")) 
  parser.add_argument("--test",type=str,default=os.environ.get("SM_test"))
  parser.add_argument("--train-file",type=str,default="train.csv")
  parser.add_argument("--test-file",type=str,default="test.csv")

  args, _ = parser.parse_known_args()

  train_df = pd.read_csv(os.path.join(args.train,args.train_file))
  test_df = pd.read_csv(os.path.join(args.test,args.test_file))

  x_train = train_df.drop("price_range",axis=1)
  y_train = train_df["price_range"] 
  x_test = test_df.drop("price_range",axis=1)
  y_test = test_df["price_range"] 

  print("x_train shape : ",x_train.shape)
  print("x_test shape : ",x_test.shape)

  # training : 

  model = RandomForestClassifier(n_estimators=args.n_estimators,random_state=args.random_state,
                                 max_depth=args.max_depth,n_jobs=args.n_jobs)
  model.fit(x_train,y_train)

  model_path = os.path.join(args.model_dir,"model.joblib")
  # joblib - just like pickle - serialized format
  joblib.dump(model,model_path)

  # prediction : 
  y_pred = model.predict(x_test)
  test_accuracy = accuracy_score(y_test,y_pred)
  print("test accuracy : ",test_accuracy)
  print(classification_report(y_test,y_pred))
  print()
  print(confusion_matrix(y_test,y_pred))



