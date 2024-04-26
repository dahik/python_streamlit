

# score detecting code starts


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#tranning code start
import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras import regularizers
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from requests_html import HTMLSession, PyQuery as pq

def predictionBasedOnNews():
    

    st.title('Stock & Forex Prediction Based On News')

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','USD/EUR','USD/AED','USD/CAD','USD/GBP')

    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    csvFilePath_feat = ''
    df = ''
    csvFilePath = ''
    outputCsvPath = ''
    

    if selected_stock=='GOOG':
        csvFilePath_feat = './data/stocks2023/goog_feat.csv'
        df = pd.read_csv(csvFilePath_feat)
        csvFilePath = './data/stocks2023/goog.csv'
        outputCsvPath = './data/stocks2023/goog_feat.csv'
    elif selected_stock=='AAPL':
        csvFilePath_feat = './data/stocks2023/aapl_feat.csv'
        df = pd.read_csv(csvFilePath_feat)
        csvFilePath = './data/stocks2023/aapl.csv'
        outputCsvPath = './data/stocks2023/aapl_feat.csv'
    elif selected_stock=='MSFT':
        csvFilePath_feat = './data/stocks2023/msft_feat.csv'
        df = pd.read_csv(csvFilePath_feat)
        csvFilePath = './data/stocks2023/msft.csv'
        outputCsvPath = './data/stocks2023/msft_feat.csv'
    elif selected_stock=='GME':
        csvFilePath_feat = './data/stocks2023/gme_feat.csv'
        df = pd.read_csv(csvFilePath_feat)
        csvFilePath = './data/stocks2023/gme.csv'
        outputCsvPath = './data/stocks2023/gme_feat.csv'
    elif selected_stock=='USD/EUR':
        csvFilePath_feat = './data/stocks2023/usdeur_feat.csv'
        df = pd.read_csv(csvFilePath_feat)
        csvFilePath = './data/stocks2023/usdeur.csv'
        outputCsvPath = './data/stocks2023/usdeur_feat.csv' 
    elif selected_stock=='USD/CAD':
        csvFilePath_feat = './data/stocks2023/usdcad_feat.csv'
        df = pd.read_csv(csvFilePath_feat)
        csvFilePath = './data/stocks2023/usdcad.csv'
        outputCsvPath = './data/stocks2023/usdcad_feat.csv' 
    elif selected_stock=='USD/AED':
        csvFilePath_feat = './data/stocks2023/usdaed_feat.csv'
        df = pd.read_csv(csvFilePath_feat)
        csvFilePath = './data/stocks2023/usdaed.csv'
        outputCsvPath = './data/stocks2023/usdaed_feat.csv'
    elif selected_stock=='USD/GBP':
        csvFilePath_feat = './data/stocks2023/usdgbp_feat.csv'
        df = pd.read_csv(csvFilePath_feat)
        csvFilePath = './data/stocks2023/usdgbp.csv'
        outputCsvPath = './data/stocks2023/usdgbp_feat.csv'
    #fetching data codes start
    # Function to fetch news articles
    def fetch_news(date):
        session = HTMLSession()
        url = f'https://www.dawn.com/archive/latest-news/{date}'
        r = session.get(url)
        articles = r.html.find('article')
        session.close()
        return articles

    # Define start and end dates
    end_date = datetime.now().date()
    start_date = datetime(2024, 1, 1)

    # Define the increment for each iteration
    dt = timedelta(days=1)

    # Initialize an empty dictionary to store all data
    all_data = {}

    # Open the existing JSON file and load its content
    existing_data = {}
    try:
        with open('./data/news2023/headlines.json', 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        pass  # If the file doesn't exist, existing_data remains an empty dictionary

    # Find the last date in the existing JSON data
    last_date = max(existing_data.keys(), default=start_date.strftime('%Y-%m-%d'))

    # Convert the last date to a datetime object
    last_date = datetime.strptime(last_date, '%Y-%m-%d').date()

    # If the last date is not the current date, fetch news articles for the dates after the last date
        # Loop through each date from the last date to the current date
    # If the last date is not the current date, fetch news articles for the dates after the last date
    if last_date != end_date:
        # Loop through each date from the last date to the current date
        for date in (last_date + timedelta(days=i) for i in range((end_date - last_date).days + 1)):
            date_str = date.strftime('%Y-%m-%d')
            print(f"Fetching news for date: {date_str}")
            articles = fetch_news(date_str)

            # Initialize an empty list to store articles for the current date
            all_data[date_str] = []

            # Append articles to the list for the current date
            for article in articles:
                t = pq(article.html)
                heading_text = t('h2.story__title a.story__link').text()
                span_id = t('span').eq(0).attr('id')
                label = span_id.lower() if span_id is not None else None
                if len(heading_text) > 0 and label in ["business", "pakistan"]:
                    all_data[date_str].append({
                        "heading": heading_text,
                        "label": label,
                    })


    # Merge the new data with the existing data
    existing_data.update(all_data)

    # Write the merged data to a new JSON file
    with open('./data/news2023/combined_headlines.json', 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False)

    # fetching data codes ends    
    # hbl to hbl feat code start
    with open('./data/news2023/daily_scores.json', 'r') as file:
        sentimentScores = json.load(file)
    #endwith

    dfSentiment = pd.DataFrame(list(sentimentScores.items()), columns=['jsonDate', 'sentiment'])
    dfSentiment['date'] = pd.to_datetime(dfSentiment['jsonDate'], format='%Y-%m-%d')
    dfSentiment.head()

    dfCsv = pd.read_csv(csvFilePath)
    dfCsv.head()

    dfCsv['date'] = pd.to_datetime(dfCsv['Date'], format='%m/%d/%Y')
    dfSentiment['date'] = pd.to_datetime(dfSentiment['date'])

    # Merge the DataFrames with left join to keep all dates from dfCsv
    dfMerged = pd.merge(dfCsv, dfSentiment, on='date', how='left')

    # Fill NaN values in 'sentiment' column with a default value (e.g., 0)
    dfMerged['sentiment'].fillna(0, inplace=True)

    # Split 'sentiment' column into multiple columns
    sentiment_columns = dfMerged['sentiment'].apply(lambda x: pd.Series(x) if isinstance(x, list) else pd.Series([0]*5))
    sentiment_columns.columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

    # Concatenate the new columns with dfMerged
    dfMerged = pd.concat([dfMerged, sentiment_columns], axis=1)

    # Drop unnecessary columns
    dfMerged.drop(columns=['Symbol', 'date', 'sentiment', 'jsonDate'], inplace=True)

    # Display the resulting DataFrame
    dfMerged.head()

    dfMerged.to_csv(outputCsvPath, index=False)

    dfMerged[['Close']].plot()
    # plt.show()
    # hbl to hbl feat code end

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification


    modelName = "nlptown/bert-base-multilingual-uncased-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    tokenizer = AutoTokenizer.from_pretrained(modelName)


    def calculateDailySentiment(headlines):
        texts = [headline['heading'] for headline in headlines]
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512, return_attention_mask=True)
        outputs = model(**inputs)
        logits = outputs.logits
        scores = logits.softmax(dim=1)
        averageScore = scores.mean(dim=0).tolist()
        return averageScore
    #enddef

    def analyzeAndSaveSentiment(inputFile, outputFile):
        with open(inputFile, 'r', errors="ignore") as file:
            data = json.load(file)

        # Load existing sentiment scores from the output file
        try:
            with open(outputFile, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = {}

        result = {}

        for date, headlines in data.items():
            # Check if sentiment score for this date already exists
            if date in existing_data:
                # Skip calculation if sentiment score already exists
                result[date] = existing_data[date]
            else:
                # Calculate sentiment score
                averageScore = calculateDailySentiment(headlines)
                print(f"{date} > {averageScore}")
                result[date] = averageScore

        # Update existing data with new scores
        existing_data.update(result)

        # Write updated data to output file
        with open(outputFile, 'w') as file:
            json.dump(existing_data, file, indent=2)

    inputJsonFile = './data/news2023/combined_headlines.json'
    outputJsonFile = './data/news2023/daily_scores.json'

    analyzeAndSaveSentiment(inputJsonFile, outputJsonFile)

    with open('./data/news2023/daily_scores.json', 'r') as file:
        sentimentScores = json.load(file)
    #endwith

    dfSentiment = pd.DataFrame(list(sentimentScores.items()), columns=['jsonDate', 'sentiment'])
    dfSentiment['date'] = pd.to_datetime(dfSentiment['jsonDate'], format='%Y-%m-%d')
    dfSentiment.head()

    csvFilePath_scores = csvFilePath
    dfCsv_score = pd.read_csv(csvFilePath_scores)
    dfCsv_score.head()

    dfCsv_score['date'] = pd.to_datetime(dfCsv_score['Date'], format='%m/%d/%Y')
    dfSentiment['date'] = pd.to_datetime(dfSentiment['date'])

    # Merge the DataFrames with left join to keep all dates from dfCsv_score
    dfMerged_score = pd.merge(dfCsv_score, dfSentiment, on='date', how='left')

    # Fill NaN values in 'sentiment' column with a default value (e.g., 0)
    dfMerged_score['sentiment'].fillna(0, inplace=True)

    # Split 'sentiment' column into multiple columns
    sentiment_columns = dfMerged_score['sentiment'].apply(lambda x: pd.Series(x) if isinstance(x, list) else pd.Series([0]*5))
    sentiment_columns.columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

    # Concatenate the new columns with dfMerged_score
    dfMerged_score = pd.concat([dfMerged_score, sentiment_columns], axis=1)

    # Drop unnecessary columns
    dfMerged_score.drop(columns=['Symbol', 'date', 'sentiment', 'jsonDate'], inplace=True)

    # Display the resulting DataFrame
    dfMerged_score.head()

    outputCsvPath = csvFilePath_feat
    dfMerged_score.to_csv(outputCsvPath, index=False)

    dfMerged_score[['Close']].plot()
    # # plt.show()
    #hbl to hbl feat code end

                 


    def createDataset(dataset, target, lookBack=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - lookBack):
            a = dataset[i:(i + lookBack), :]
            dataX.append(a)
            dataY.append(target[i + lookBack])
        return np.array(dataX), np.array(dataY)

    features = df.drop(['Date', 'Close'], axis=1).values
    target = df['Close'].values

    scalerFeatures = MinMaxScaler(feature_range=(0, 1))
    scalerTarget = MinMaxScaler(feature_range=(0, 1))

    featuresScaled = scalerFeatures.fit_transform(features)
    targetScaled = scalerTarget.fit_transform(target.reshape(-1, 1))

    lookBack = 3
    X, y = createDataset(featuresScaled, targetScaled, lookBack)

    trainSize = int(len(X) * 0.8)
    testSize = len(X) - trainSize
    trainX, testX = X[0:trainSize, :], X[trainSize:len(X), :]
    trainY, testY = y[0:trainSize], y[trainSize:len(y)]

    trainX = np.reshape(trainX, (trainX.shape[0], lookBack, trainX.shape[2]))
    testX = np.reshape(testX, (testX.shape[0], lookBack, testX.shape[2]))

    batchSize = 1
    epoch = 20
    neurons = 100
    dropout = 0.6

    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, activation='tanh', input_shape=(lookBack, features.shape[1])))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=True, activation='tanh'))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, activation='tanh'))
    model.add(Dropout(dropout))

    model.add(Dense(units=1, activation='linear', activity_regularizer=regularizers.l1(0.00001)))
    model.add(Activation('tanh'))

    model.compile(loss='mean_squared_error' , optimizer='RMSprop')

    model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=1, validation_split=0.2)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredictInv = scalerTarget.inverse_transform(trainPredict)
    trainYInv = scalerTarget.inverse_transform(np.reshape(trainY, (trainY.shape[0], 1)))
    testPredictInv = scalerTarget.inverse_transform(testPredict)
    testYInv = scalerTarget.inverse_transform(np.reshape(testY, (testY.shape[0], 1)))

    train_nan_indices = np.isnan(trainYInv) | np.isnan(trainPredictInv)
    test_nan_indices = np.isnan(testYInv) | np.isnan(testPredictInv)

    trainYInv[train_nan_indices] = 0
    trainPredictInv[train_nan_indices] = 0
    testYInv[test_nan_indices] = 0
    testPredictInv[test_nan_indices] = 0

    trainScore = np.sqrt(mean_squared_error(trainYInv[:, 0], trainPredictInv[:, 0]))
    testScore = np.sqrt(mean_squared_error(testYInv[:, 0], testPredictInv[:, 0]))

    trainAccuracy = 100 - (trainScore / np.mean(trainYInv) * 100)
    testAccuracy = 100 - (testScore / np.mean(testYInv) * 100)

    st.write(f'Training Accuracy: {trainAccuracy:.2f}%')
    st.write(f'Testing Accuracy: {testAccuracy:.2f}%')

    dates = df['Date'].values.tolist()

    trainY_list = trainY.tolist()
    testY_list = testY.tolist()

    num_ticks = 8
    sampleInterval = math.ceil(len(dates) / num_ticks)
    sampledIndices = list(range(0, len(dates), sampleInterval))
    sampledDates_str = [str(dates[idx]) for idx in sampledIndices if str(dates[idx]) != 'nan']

    trainY_tuples = [tuple(item) for item in trainY_list]
    testY_tuples = [tuple(item) for item in testY_list]
    common_dates = set(trainY_tuples).intersection(testY_tuples)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dates[:len(trainY)], trainY, label='Actual Train')
    ax.plot(dates[:len(trainPredict)], trainPredict, label='Predicted Train')

    ax.plot(dates[len(trainY):len(trainY) + len(testY)], testY, label='Actual Test')
    ax.plot(dates[len(trainPredict):len(trainPredict) + len(testPredict)], testPredict, label='Predicted Test')
    sampledDates_dt = [datetime.strptime(date_str, '%m/%d/%Y') for date_str in sampledDates_str]

    ax.set_xticks(sampledIndices)
    ax.set_xticklabels(sampledDates_str, rotation=90)
    ax.legend()
    st.pyplot(fig)







