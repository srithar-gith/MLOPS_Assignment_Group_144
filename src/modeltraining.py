import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def main():
    # -------- 1. Load Data --------python- modeltraining.py
    #iris = load_iris()
    file_path = 'data/Iris_dataset.csv'
    df = pd.read_csv(file_path)

    # 
    # -------- 2. Data Preprocessing --------

    #Drop any missing rows (optional, if your file is clean, this has no effect)
    df.dropna(inplace=True)

    #Encode target labels
    label_encoder = LabelEncoder()

    df['target'] = label_encoder.fit_transform(df[' Species'])

    # Split features and labels
    
    X = df[[" SepalLengthCm", " SepalWidthCm", " PetalLengthCm", " PetalWidthCm"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- 3. Configure MLflow --------
    mlflow.set_tracking_uri("file:./mlruns")  # Local folder tracking
    mlflow.set_experiment("Iris_dataset_Classification")
    mlflow.sklearn.autolog()
    

    #mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, log_system_metrics=True)

    models_results = {}

    # -------- 4. Train & Track Models --------
    

    ## Logistic Regression
    with mlflow.start_run(run_name="Logistic_Regression") as run:
        
            lr = LogisticRegression(max_iter=200)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_param("max_iter", 200)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(lr, "model")
           
           
            models_results["Logistic_Regression"] = (acc, mlflow.active_run().info.run_id)
        
    ## Random Forest
    with mlflow.start_run(run_name="Random_Forest") as run:

        
            rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 4)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(rf, "model")
            #mlflow.system_metrics.log_system_metrics()
            models_results["Random_Forest"] = (acc, mlflow.active_run().info.run_id)
        
   

    # -------- 5. Select Best Model & Register --------
    best_model_name = max(models_results, key=lambda k: models_results[k][0])
    best_accuracy, best_run_id = models_results[best_model_name]

    print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy}")

    model_uri = f"runs:/{best_run_id}/model"
    model_registered_name = "Iris_Best_Model"

    try:
        mlflow.register_model(model_uri=model_uri, name=model_registered_name)
        print(f"Model '{best_model_name}' registered as '{model_registered_name}' in MLflow!")
    except Exception as e:
        print("Model Registry requires MLflow server with backend DB.")
        print("Skipping registration. Error:", e)

    # Load the best model from MLflow
    best_model = mlflow.sklearn.load_model(model_uri)

    # Save it using joblib
    joblib.dump(best_model, f"model/{best_model_name}_best_model.pkl")
    print(f"💾 Best model saved as '{best_model_name}_best_model.pkl'")

if __name__ == "__main__":
    main()
