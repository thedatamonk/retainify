# FastAPI app containing inference logic
# Inference endpoint will load data from CSV
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from utils.file_utils import check_valid_csv
from utils.data_utils import process_test_data
from utils.model_utils import test_model
from io import StringIO
import pandas as pd
import logging
from datetime import datetime
import os
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


app = FastAPI(title="Model Inference Services")

logger = logging.getLogger("inference_logger")

# Set the MLflow tracking URI and model name
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'  # Adjust according to your setup
MODEL_NAME = "ChurnPredictorModel"  # The name you registered your model under in MLflow



@app.post("/predict")
async def generate_predictions(file: UploadFile = File(...)):
    logger.info(f'Received file: {file.filename}')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


    try:
        if check_valid_csv(file):

            logger.info(f'Uploaded file is a valid csv file')

            # Load the raw data from the file
            contents = await file.read()

            string_buffer = StringIO(contents.decode("utf-8"))

            data = pd.read_csv(string_buffer)
            logger.info(f"Data from {file.filename} loaded successfully in a Pandas dataframe")

            # Data cleaning and feature selection
            processed_data = await process_test_data(data)

            # Store processed data in local storage
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            INFERENCE_BASE_DIR = "./data/inference"
            if not os.path.exists(INFERENCE_BASE_DIR):
                os.makedirs(INFERENCE_BASE_DIR)

            processed_filepath = os.path.join(INFERENCE_BASE_DIR, f"processed_data_{timestamp}.csv")
            processed_data.to_csv(processed_filepath, index=False)

            logger.info(f"Processed data stored successfully in {processed_filepath}")
            logger.info(f"\n{processed_data.head()}")

            predictions = await test_model(model_name=MODEL_NAME, data=processed_data)
            logger.info(f"Predictions generated successfully")
            
            return predictions
        else:
            raise HTTPException(status_code=422, detail="File format not supported. Please upload a CSV file.")
    except Exception as e:
        logger.error(f"Model predictions could not be generated due to error: {e}")
        raise HTTPException(status_code=500, detail="Model inference could not finish successfully.")



