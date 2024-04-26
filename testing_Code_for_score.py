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

# def analyzeAndSaveSentiment(inputFile, outputFile):
#     with open(inputFile, 'r', errors="ignore") as file:
#         data = json.load(file)
#     #endwith

#     result = {}

#     for date, headlines in data.items():
#         averageScore = calculateDailySentiment(headlines)
#         print(f"{date} > {averageScore}")
#         result[date] = averageScore
#     #endfor

#     with open(outputFile, 'w') as outputFile:
#         json.dump(result, outputFile, indent=2)
#     #endwith
# #enddef

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


import pandas as pd
import matplotlib.pyplot as plt

with open('./data/news2023/daily_scores.json', 'r') as file:
    sentimentScores = json.load(file)
#endwith

dfSentiment = pd.DataFrame(list(sentimentScores.items()), columns=['jsonDate', 'sentiment'])
dfSentiment['date'] = pd.to_datetime(dfSentiment['jsonDate'], format='%Y-%m-%d')
dfSentiment.head()

csvFilePath = './data/stocks2023/hbl.csv'
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

outputCsvPath = './data/stocks2023/hbl_feat.csv'
dfMerged.to_csv(outputCsvPath, index=False)

dfMerged[['Close']].plot()
# # plt.show()
#hbl to hbl feat code end
