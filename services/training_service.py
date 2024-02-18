# FastAPI app containing training logic

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from utils.file_utils import check_valid_csv
from utils.data_utils import process_data
from utils.model_utils import initiate_training
from io import StringIO
import pandas as pd
import logging
from datetime import datetime
import os
import mlflow
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


app = FastAPI(title="Model Training Services")

logger = logging.getLogger("training_logger")

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    logger.info(f'Received file: {file.filename}')

    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    try:
        if check_valid_csv(file):

            logger.info(f'Uploaded file is a valid csv file')

            # Load the raw data from the file
            contents = await file.read()

            string_buffer = StringIO(contents.decode("utf-8"))

            data = pd.read_csv(string_buffer)
            logger.info(f"Data from {file.filename} loaded successfully in a Pandas dataframe")

            with mlflow.start_run(experiment_id="168527006646638648") as run:
                # Tag the run
                mlflow.set_tag('stage', 'dev')

                # Data cleaning and feature selection
                processed_data = await process_data(data)

                # Store processed data in local storage
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

                if not os.path.exists("./data"):
                    os.makedirs("./data")

                processed_filepath = os.path.join("./data", f"processed_data_{timestamp}.csv")
                processed_data.to_csv(processed_filepath, index=False)

                logger.info(f"Processed data stored successfully in {processed_filepath}")
                logger.info(f"\n{processed_data.head()}")


                # Initiate training
                model, metrics = await initiate_training(data=processed_data)

                # Log the metrics
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                # Log the model
                mlflow.sklearn.log_model(model, "model")

                # Register the model
                MODEL_NAME="ChurnPredictorModel"
                result = mlflow.register_model(
                    model_uri=f"runs:/{run.info.run_id}/model", name=MODEL_NAME
                )

                client = mlflow.tracking.MlflowClient()
                
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=result.version,
                    stage="Staging",
                )

            return JSONResponse(content="Model trained and logged successfully in MLflow")
        else:
            raise HTTPException(status_code=422, detail="File format not supported. Please upload a CSV file.")
    except Exception as e:
        logger.error(f"Model training could not finish due to error: {e}")
        raise HTTPException(status_code=500, detail="Model training could not finish successfully.")

