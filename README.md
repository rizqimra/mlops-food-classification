# mlops-food-classification
Final Project of Machine Learning Operations (MLOps)

## Project Guide
1. **Clone the Project**

   To clone the project, run the following command in your terminal:
   ```bash
    git clone https://github.com/rizqimra/mlops-food-classification.git

    cd mlops-food-classification
   ```
   Then create a new virtual environment using these commands:
   ```conda
    conda create -n "myenv" python=3.9

    conda activate myenv

		pip install -r requirements.txt
   ```
   
3. **Create `.env` File**
   
   Create your own .env file configuration, for example:
   ```.env		
		MLFLOW_ARTIFACT=mlflow
		MLFLOW_TRACKING_URI=http://localhost:5000
		MLFLOW_S3_ENDPOINT_URL=http://minio:9000
		MLFLOW_TRACKING_USERNAME=minio
		MLFLOW_TRACKING_PASSWORD=minio123
		AWS_ACCESS_KEY_ID=minio
		AWS_SECRET_ACCESS_KEY=minio123
		MINIO_URL=http://minio:9000
		MINIO_ACCESS_KEY=minio
		MINIO_SECRET_KEY=minio123
   ```

4. **Access MinIO UI**:
	- Download MinIO.
	- Go to [http://localhost:9000](http://localhost:9000) in your browser.
	- Log in with the MinIO credentials defined in `.env` file or using the default credentials provided by minio.
	- Create ***models*** and ***data*** buckets for storing models and data. Enable versioning for ***data*** bucket.
	- Upload the data provided in the repository to MinIO.

5. **Track Experiments and Models with MLflow**:
	- Run `mlflow ui` command in your terminal.
	- Go to [http://localhost:5000](http://localhost:5000) in your web browser.
	- Log in with the MLflow credentials defined in `.env` file.
	- Run the `train.py` script, check the experiment results in the MLflow UI after the training is completed.

6. **Run the App in Streamlit**:
	- Add a RUN_ID configuration in your .env file. You can check the RUN_ID from the previous experiment in MLflow UI.
	- Run `streamlit run src/streamlit.py` command in your terminal.
	- Go to [http://localhost:8501](http://localhost:8501) in your web browser.
	- Try to predict an image in the streamlit app.
