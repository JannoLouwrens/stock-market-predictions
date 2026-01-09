# Contains keyword tesla MSE: 66.45
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from flair.models import TextClassifier
from flair.data import Sentence
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Assuming these files exist in the same directory as your script
stock_price_data = pd.read_csv('/content/drive/MyDrive/tesla_stock_price_data.csv')
financial_news_data = pd.read_csv('/content/drive/MyDrive/tesla_financial_news_data.csv')

# Assuming 'date' is the common date column in both datasets
# Pre-processing: Convert 'date' to datetime if not already
stock_price_data['date'] = pd.to_datetime(stock_price_data['date'])
financial_news_data['date'] = pd.to_datetime(financial_news_data['date'])
# Drop rows with NaN values in the 'text' column
financial_news_data = financial_news_data.dropna(subset=['text'])

# Filter headlines containing "Tesla"
tesla_headlines = financial_news_data[financial_news_data['text'].str.contains('Tesla', case=False)]

# Initialize the VADER sentiment analyzer and Flair text classifier
vader_analyzer = SentimentIntensityAnalyzer()
flair_classifier = TextClassifier.load('sentiment')

# Define function to get VADER sentiment scores for a given text
def get_vader_sentiment_score(text):
    if isinstance(text, str):  # Check if text is a string
        sentiment = vader_analyzer.polarity_scores(text)['compound']
        return sentiment
    else:
        return 0.0  # Or handle it in a way that makes sense for your analysis

# Define a function to get Flair sentiment scores for a given text
def get_flair_sentiment_score(text):
    if isinstance(text, str):  # Check if text is a string
        sentence = Sentence(text)
        flair_classifier.predict(sentence)
        if sentence.labels:
            # At least one label is predicted
            label_dict = {label.value: label.score for label in sentence.labels}
            positive_prob = label_dict.get('POSITIVE', 0.0)
            negative_prob = label_dict.get('NEGATIVE', 0.0)
            sentiment_score = positive_prob - negative_prob
            return sentiment_score
        else:
            return 0.0  # No label predicted
    else:
        return 0.0  # Or handle it in a way that makes sense for your analysis

# Define a function to get TextBlob sentiment scores for a given text
def get_textblob_sentiment_score(text):
    if isinstance(text, str):  # Check if text is a string
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        return sentiment_score
    else:
        return 0.0  # Or handle it in a way that makes sense for your analysis

# Apply the functions to each row in the filtered financial news data
tesla_headlines['flair_sentiment_score'] = tesla_headlines['text'].apply(get_flair_sentiment_score)
tesla_headlines['textblob_sentiment_score'] = tesla_headlines['text'].apply(get_textblob_sentiment_score)
tesla_headlines['vader_sentiment_score'] = tesla_headlines['text'].apply(get_vader_sentiment_score)

# Aggregate the sentiment scores by date to get the average sentiment of that day's Tesla news
average_sentiment_scores_tesla = tesla_headlines.groupby('date').agg({
    'flair_sentiment_score': 'mean',
    'textblob_sentiment_score': 'mean',
    'vader_sentiment_score': 'mean'
}).reset_index()

# Merge datasets based on the 'date' column
merged_data_tesla = pd.merge(stock_price_data, average_sentiment_scores_tesla, on='date', how='left')

# Drop rows with missing values in any column
merged_data_tesla.dropna(inplace=True)

# Create lag features for closing prices and sentiment scores
merged_data_tesla['lag_closing_price_1'] = merged_data_tesla['closing_price'].shift(1)
merged_data_tesla['lag_flair_sentiment_score_1'] = merged_data_tesla['flair_sentiment_score'].shift(3)
merged_data_tesla['lag_textblob_sentiment_score_1'] = merged_data_tesla['textblob_sentiment_score'].shift(3)
merged_data_tesla['lag_vader_sentiment_score_1'] = merged_data_tesla['vader_sentiment_score'].shift(3)

# Drop rows with missing values introduced by lag features
merged_data_tesla.dropna(inplace=True)

# Create a feature set for Tesla headlines
feature_set_tesla = pd.DataFrame({
    'date': merged_data_tesla['date'],
    'closing_price': merged_data_tesla['closing_price'],
    'sentiment_flair_score': merged_data_tesla['flair_sentiment_score'],
    'sentiment_textblob_score': merged_data_tesla['textblob_sentiment_score'],
    'sentiment_vader_score': merged_data_tesla['vader_sentiment_score'],
    'lag_closing_price_1': merged_data_tesla['lag_closing_price_1'],
    'lag_flair_sentiment_score_1': merged_data_tesla['lag_flair_sentiment_score_1'],
    'lag_textblob_sentiment_score_1': merged_data_tesla['lag_textblob_sentiment_score_1'],
    'lag_vader_sentiment_score_1': merged_data_tesla['lag_vader_sentiment_score_1']
})

# Split the feature set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_set_tesla.drop(['date', 'closing_price',
                            'sentiment_flair_score', 'sentiment_textblob_score', 'sentiment_vader_score'], axis=1),
    feature_set_tesla['closing_price'],
    test_size=0.1,
    random_state=42,
    shuffle=False
)
print("X_train:")
print(X_train)

print("\ny_train:")
print(y_train)

print("\nX_test:")
print(X_test)

print("\ny_test:")
print(y_test)
# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model on the scaled training set
model = MLPRegressor(
    hidden_layer_sizes=(500, 500, 500),  # Adjust the number of neurons as needed
    max_iter=10000,
    learning_rate='constant',
    random_state=43
)
model.fit(X_train_scaled, y_train)

# Evaluate the model on the scaled test set
y_pred_test = model.predict(X_test_scaled)

# Calculate and print the mean squared error on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error on Test Set: {mse_test}")

# Print actual values and predicted values for the test set
actual_vs_predicted_with_data = pd.DataFrame({
    'Date': X_test.index,  # Assuming 'date' is the index
    'Actual Closing Price': y_test,
    'Predicted Closing Price': y_pred_test,
    'Sentiment Flair Score': merged_data_tesla.loc[X_test.index, 'flair_sentiment_score'],
    'Sentiment TextBlob Score': merged_data_tesla.loc[X_test.index, 'textblob_sentiment_score'],
    'Sentiment VADER Score': merged_data_tesla.loc[X_test.index, 'vader_sentiment_score'],
    'Actual News Headline': financial_news_data.loc[X_test.index, 'text'], # Fetching news headlines from the original data
})

# Sort the index before plotting
actual_vs_predicted_with_data = actual_vs_predicted_with_data.sort_values('Date')

print("\nActual vs. Predicted Values on Test Set with Additional Data:")
print(actual_vs_predicted_with_data)

# Sort the index before plotting
actual_vs_predicted_with_data = actual_vs_predicted_with_data.sort_values('Date')

# Plot a graph of actual vs. predicted closing prices
plt.plot(actual_vs_predicted_with_data['Date'], actual_vs_predicted_with_data['Actual Closing Price'],
         label='Actual Closing Price', marker='o')
plt.plot(actual_vs_predicted_with_data['Date'], actual_vs_predicted_with_data['Predicted Closing Price'],
         label='Predicted Closing Price', marker='x')
plt.legend()
plt.xlabel('Date (Test Set)')
plt.ylabel('Closing price')
plt.title('Actual vs. Predicted Closing Prices on Test Set')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()

# Create a DataFrame with actual values, sentiment scores, news headlines, and dates for the training set
result_df_train = pd.DataFrame({
    'Date': merged_data_tesla.loc[X_train.index, 'date'],
    'Actual Closing Price': y_train,
    'Sentiment Flair Score': merged_data_tesla.loc[X_train.index, 'flair_sentiment_score'],
    'Sentiment TextBlob Score': merged_data_tesla.loc[X_train.index, 'textblob_sentiment_score'],
    'Sentiment VADER Score': merged_data_tesla.loc[X_train.index, 'vader_sentiment_score'],
    'Actual News Headline': financial_news_data.loc[X_train.index, 'text']
})

# Step 1: Generate future dates
future_dates = pd.date_range(start=merged_data_tesla['date'].max() + pd.Timedelta(days=1), periods=2, freq='D')

# Step 2: Create lag features for sentiment scores for future dates
future_lag_features = pd.DataFrame({
    'lag_closing_price_1': merged_data_tesla['closing_price'].tail(2).values,
    'lag_flair_sentiment_score_1': merged_data_tesla['flair_sentiment_score'].tail(2).values,
    'lag_textblob_sentiment_score_1': merged_data_tesla['textblob_sentiment_score'].tail(2).values,
    'lag_vader_sentiment_score_1': merged_data_tesla['vader_sentiment_score'].tail(2).values,
}, index=future_dates)

# Step 3: Predict closing prices for future dates
future_lag_features_scaled = scaler.transform(future_lag_features)
future_predictions = model.predict(future_lag_features_scaled)

# Step 4: Combine predictions with existing data
future_predictions_df = pd.DataFrame({
    'date': future_dates,
    'predicted_closing_price': future_predictions,
    'lag_closing_price_1': future_lag_features['lag_closing_price_1'],
    'lag_flair_sentiment_score_1': future_lag_features['lag_flair_sentiment_score_1'],
    'lag_textblob_sentiment_score_1': future_lag_features['lag_textblob_sentiment_score_1'],
    'lag_vader_sentiment_score_1': future_lag_features['lag_vader_sentiment_score_1'],
})

merged_data_tesla = pd.concat([merged_data_tesla, future_predictions_df], ignore_index=True)

# Display the combined DataFrame with future predictions
print("\nCombined Actual vs. Predicted Values with Additional Data (including future predictions):")
print(merged_data_tesla)

# Print predicted closing prices for the next two days
print("\nPredicted Closing Prices for the Next Two Days:")
print(f"Tomorrow: {future_predictions[0]}")
print(f"Day after tomorrow: {future_predictions[1]}")

merged_data_tesla = merged_data_tesla.reset_index(drop=True)

# Create a DataFrame with actual values, sentiment scores, news headlines, and dates for the test set
result_df_test = pd.DataFrame({
    'Date': actual_vs_predicted_with_data['Date'],
    'Actual Closing Price': y_test,
    'Predicted Closing Price': y_pred_test,
    'Sentiment Flair Score': merged_data_tesla.loc[X_test.index, 'flair_sentiment_score'],
    'Sentiment TextBlob Score': merged_data_tesla.loc[X_test.index, 'textblob_sentiment_score'],
    'Sentiment VADER Score': merged_data_tesla.loc[X_test.index, 'vader_sentiment_score'],
    'News Headline': financial_news_data.loc[X_test.index, 'text']
})

# Sort the DataFrame by date
result_df_train = result_df_train.sort_values('Date')

# Combine the training and test dataframes
result_df_combined = pd.concat([result_df_train, result_df_test], axis=0)

# Ensure 'Date' column is in datetime format
result_df_combined['Date'] = pd.to_datetime(result_df_combined['Date'], errors='coerce')

# Drop rows with missing datetime values
result_df_combined.dropna(subset=['Date'], inplace=True)

# Sort the DataFrame by date
result_df_combined = result_df_combined.sort_values('Date')

# Create a larger and wider figure
plt.figure(figsize=(15, 8))

# Plot a graph of actual closing prices for training and test set
plt.plot(result_df_combined['Date'], result_df_combined['Actual Closing Price'],
         label='Actual Closing Price', marker='o')
plt.plot(result_df_combined['Date'], result_df_combined['Predicted Closing Price'],
         label='Predicted Closing Price', marker='x')
plt.legend()

# Add predictions for the test set to the same graph
plt.scatter(result_df_combined['Date'], result_df_combined['Predicted Closing Price'],
            label='Predicted Closing Price', marker='x', color='red')

plt.xlabel('Date')
plt.ylabel('Closing price')
plt.title('Actual Closing Prices with Predictions')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

plt.show()

# Sort the DataFrame by date
result_df_test = result_df_test.sort_values('Date')

# Combine the training and test dataframes
result_df_combined = pd.concat([result_df_train, result_df_test], axis=0)

# Save the combined DataFrame to an Excel file
result_df_combined.to_excel('result_data_combined.xlsx', index=False)

# Display the combined DataFrame
print("\nCombined Actual vs. Predicted Values with Additional Data:")
print(result_df_combined)
