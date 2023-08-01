from stockify.entity.artifact_entity import DataTransformationArtifact , DataIngestionArtifact
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from scipy import stats
import tensorflow
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pickle
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact:DataTransformationArtifact,
                 data_ingestion_artifact: DataIngestionArtifact
                 ):

        self.data_transformation_artifact = data_transformation_artifact
        self.data_ingestion_artifact = data_ingestion_artifact    
    
    def LSTM_model(self, timestamp=6):
        
        csv_files = self.data_ingestion_artifact.csv_files
        pickle.dump(csv_files, open('Tech_data_path.pkl', 'wb'))
        
        def X_Y(df, timestamp):
            X, Y = [], []
            for i in range(0, len(df) - timestamp - 1):
                X.append(df[i:(i + timestamp), :])
                Y.append(df[i + timestamp, :])

            if len(df) >= timestamp:
                X.append(df[-timestamp:, :])
                Y.append(df[-1, :])

            return np.array(X), np.array(Y)
        
        def get_confidence_interval(errors, confidence=0.95):
            mean_error = np.mean(errors)
            standard_error = np.std(errors, ddof=1) / np.sqrt(len(errors))

            t_score = np.abs(stats.t.ppf((1 + confidence) / 2, len(errors) - 1))

            lower_bound = mean_error - t_score * standard_error
            upper_bound = mean_error + t_score * standard_error
            
             # Inverse transform to get prices in the original scale
            lower_bound = scaler.inverse_transform([[0, 0, 0, lower_bound]])[0, -1]
            upper_bound = scaler.inverse_transform([[0, 0, 0, upper_bound]])[0, -1]

            return round(lower_bound), round(upper_bound)
        

        def get_recommendation(model, xtest, ytest, scaler, last_day_price):
            pred = model.predict(xtest)
            
            last_sequence = xtest[-1]
            last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])

            next_day_pred = model.predict(last_sequence)
            next_day_pred_original_scale = scaler.inverse_transform(next_day_pred)
            next_day_closing_price = next_day_pred_original_scale[0, -1]

            accuracy = r2_score(ytest, pred)
            current_price = last_day_price[-1]
            
            errors = np.abs(ytest - pred)
            confidence_interval = get_confidence_interval(errors)

            recommendation = ((next_day_closing_price - current_price) / current_price) * 100

            return current_price, recommendation, accuracy, confidence_interval

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

            current_price, recommendation, accuracy,confidence_interval = get_recommendation(model, xtest, ytest, scaler, df['Close'].values)
            result_data.append({
                "stock_ticker": input_filename,
                "current_price": round(current_price,2),
                "recommendation":  f"{round(recommendation,2)}%",
                "accuracy": f"{round(accuracy,2)*100}%",
                "confidence_interval": confidence_interval
            })
            
        json.dump(result_data, open('Output/result_data.json', 'w'))
        return result_data


    def FinBert(self):
        model_name = "ProsusAI/finbert"  
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        csv_files = self.data_ingestion_artifact.news_data
        try:
            with open('Output/news_sentiment.json', 'r') as json_file:
                existing_data = json.load(json_file)
        except FileNotFoundError:
            existing_data = []
        
        for file in csv_files:
            df = pd.read_csv(file,nrows=10)
            df_array = np.array(df)
            df_list = list(df_array[:, 0])
            inputs = tokenizer(df_list, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            max_probs, max_labels = torch.max(predictions, dim=1)
            max_labels = max_labels.tolist()

            labels = ["Positive", "Negative", "Neutral"]
            max_labels = [labels[label] for label in max_labels]
            
            input_filename = os.path.basename(file).split(".")[0]
            new_data = {
                "Stock_ticker" : input_filename,
                'Headline': df_list,
                "Max_Probability_Value": max_probs.tolist(),
                "Max_Probability_Label": max_labels,
            }
            found_ticker = False
            for item in existing_data:
                if item['Stock_ticker'] == input_filename:
                    for new in new_data['Headline']:
                        if new not in item['Headline']:  # Only add if it's not already in existing headlines
                            item['Headline'].append(new)
                            item['Max_Probability_Value'].append(new_data['Max_Probability_Value'][df_list.index(new)])
                            item['Max_Probability_Label'].append(new_data['Max_Probability_Label'][df_list.index(new)])
                    found_ticker = True
                    break

            if not found_ticker:
                existing_data.append(new_data)

        with open('Output/news_sentiment.json', 'w') as json_file:
            json.dump(existing_data, json_file)

        return existing_data


