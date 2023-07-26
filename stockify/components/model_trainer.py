from stockify.exception import StockifyExpection
import sys
from stockify.logger import logging
from stockify.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact , DataIngestionArtifact
from stockify.entity.config_entity import ModelTrainerConfig
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM ,Dense , Dropout,Bidirectional
from sklearn.preprocessing import MinMaxScaler
import math
import json
from stockify.components.data_transformation import DataTransformation
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

        
class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact:DataTransformationArtifact,
                 data_ingestion_artifact: DataIngestionArtifact
                 ):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.data_ingestion_artifact = data_ingestion_artifact    
    
    def LSTM_model(self, timestamp=100):
        
        csv_files = self.data_ingestion_artifact.csv_files
        def X_Y(df, timestamp):
            X, Y = [], []
            for i in range(0, len(df) - timestamp - 1):
                X.append(df[i:(i + timestamp), :])
                Y.append(df[i + timestamp, :])

            if len(df) >= timestamp:
                X.append(df[-timestamp:, :])
                Y.append(df[-1, :])

            return np.array(X), np.array(Y)

        def get_recommendation(model, xtest, ytest, scaler, last_day_price):
            pred = model.predict(xtest)
            
            last_sequence = xtest[-1]
            last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])

            next_day_pred = model.predict(last_sequence)
            next_day_pred_original_scale = scaler.inverse_transform(next_day_pred)
            next_day_closing_price = next_day_pred_original_scale[0, -1]

            accuracy = r2_score(ytest, pred)
            current_price = last_day_price[-1]

            recommendation = ((next_day_closing_price - current_price) / current_price) * 100

            return current_price, recommendation, accuracy

        result_data = []
        for file_path in csv_files:
            
            input_filename = os.path.basename(file_path).split(".")[0]
            
            df = pd.read_csv(file_path)
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.date

            original_data = df[['Open', 'High', 'Low',"Close"]].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(original_data)

            train_size = int(len(scaled_data) * 0.70)
            xtrain, ytrain = X_Y(scaled_data[0:train_size], timestamp)
            xtest, ytest = X_Y(scaled_data[train_size:], timestamp)

            xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 4)
            xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 4)

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(xtrain.shape[1], 4)))
            model.add(LSTM(50, return_sequences=True))
            model.add(LSTM(50))
            model.add(Dense(4))
            model.compile(loss='mse', optimizer='adam')

            history = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=20, batch_size=64, verbose=1)

            current_price, recommendation, accuracy = get_recommendation(model, xtest, ytest, scaler, df['Close'].values)
            result_data.append({
                "stock_ticker": input_filename,
                "current_price": round(current_price,2),
                "recommendation":  f"{round(recommendation,2)}%",
                "accuracy": f"{round(accuracy,2)*100}%"
            })
        json.dump(result_data, open('Output/result_data.json', 'w'))
        return result_data


    def FinBert(self):
        all_output_dataframes = []
        model_name = "ProsusAI/finbert"  

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        csv_files = self.data_ingestion_artifact.news_data
        # print("csv_files --> ", csv_files)
        for file in csv_files:
            df = pd.read_csv(file, nrows=5)
            df_array = np.array(df)
            df_list = list(df_array[:, 0])

            inputs = tokenizer(df_list, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Tweet with highest probability and corresponding label
            max_probs, max_labels = torch.max(predictions, dim=1)
            max_labels = max_labels.tolist()

            # Convert label indices to label names (positive, negative, neutral)
            labels = ["Positive", "Negative", "Neutral"]
            max_labels = [labels[label] for label in max_labels]
            
            input_filename = os.path.basename(file).split(".")[0]

            table = {
                "Stock_ticker" : input_filename,
                'Headline': df_list,
                "Max_Probability_Value": max_probs.tolist(),
                "Max_Probability_Label": max_labels,
            }
            
            # df_output = pd.DataFrame(table, columns=["Headline", "Max_Probability_Value", "Max_Probability_Label",])
            all_output_dataframes.append(table)

            # Save the output DataFrame into a separate pickle file for each company
            
            # file_name_without_extension = input_filename.split(".")[0]
            # output_filename = f"saved_model/{file_name_without_extension}_sentiment.json"
            # with open(output_filename, 'w') as f:
            #     json.dump(df_output, f)
                
        json.dump(all_output_dataframes, open(f'Output/news_sentiment.json', 'w'))

        return all_output_dataframes
    
        

