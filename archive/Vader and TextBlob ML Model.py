import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from flair.models import TextClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob  # Import TextBlob

# Load the historical stock price data and financial news data
stock_price_data = pd.read_csv('/stock_price_data.csv')
financial_news_data = pd.read_csv('/financial_news_data.csv')

# Assuming 'date' is the common date column in both datasets
# Merge datasets based on the 'date' column
merged_data = pd.merge(stock_price_data, financial_news_data, on='date', how='left')

# Initialize the sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()

# Define a function to get Vader sentiment scores for a given text
def get_vader_sentiment_score(text):
    if isinstance(text, str):  # Check if text is a string
        sentiment = vader_analyzer.polarity_scores(text)['compound']
        return sentiment
    else:
        return 0.0  # Or handle it in a way that makes sense for your analysis

# Apply the function to each row in the financial news data
merged_data['vader_sentiment_score'] = merged_data['text'].apply(get_vader_sentiment_score)

# Define a function to get TextBlob sentiment scores for a given text
def get_textblob_sentiment_score(text):
    if isinstance(text, str):  # Check if text is a string
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    else:
        return 0.0  # Or handle it in a way that makes sense for your analysis

# Apply the function to each row in the financial news data
merged_data['textblob_sentiment_score'] = merged_data['text'].apply(get_textblob_sentiment_score)

# Drop rows with missing values in any column
merged_data.dropna(inplace=True)

# Create lag features for closing prices and sentiment scores
merged_data['lag_closing_price_1'] = merged_data['closing_price'].shift(1)
merged_data['lag_sentiment_score_1'] = merged_data['vader_sentiment_score'].shift(1)
merged_data['lag_textblob_sentiment_score_1'] = merged_data['textblob_sentiment_score'].shift(1)

# Drop rows with missing values introduced by lag features
merged_data.dropna(inplace=True)

# Create a feature set
feature_set = pd.DataFrame({
    'date': merged_data['date'],
    'closing_price': merged_data['closing_price'],
    'sentiment_vader_score': merged_data['vader_sentiment_score'],
    'sentiment_textblob_score': merged_data['textblob_sentiment_score'],
    'lag_closing_price_1': merged_data['lag_closing_price_1'],
    'lag_sentiment_score_1': merged_data['lag_sentiment_score_1'],
    'lag_textblob_sentiment_score_1': merged_data['lag_textblob_sentiment_score_1']
})

# Split the feature set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_set.drop(['date', 'closing_price', 'sentiment_vader_score', 'sentiment_textblob_score'], axis=1),
    feature_set['closing_price'],
    test_size=0.1,
    random_state=42,
    shuffle=False
)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model on the scaled training set
model = MLPRegressor(max_iter=10000, learning_rate='constant', random_state=43)
model.fit(X_train_scaled, y_train)

# Evaluate the model on the scaled test set
y_pred_test = model.predict(X_test_scaled)

# Calculate and print the mean squared error on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error on Test Set: {mse_test}")

# Plot each actual and predicted stock price for the test set
plt.scatter(X_test.index, y_test, label='Actual stock prices', marker='o')
plt.scatter(X_test.index, y_pred_test, label='Predicted stock prices', marker='x')
plt.legend()
plt.xlabel('Date Index (Test Set)')
plt.ylabel('Closing price')
plt.title('Actual vs. predicted stock prices on Test Set')
plt.show()

# Print actual values and predicted values for the test set
actual_vs_predicted_with_data = pd.DataFrame({
    'Date': X_test.index,  # Assuming 'date' is the index
    'Actual Closing Price': y_test,
    'Predicted Closing Price': y_pred_test,
    'Sentiment Vader Score': merged_data.loc[X_test.index, 'vader_sentiment_score'],
    'Sentiment TextBlob Score': merged_data.loc[X_test.index, 'textblob_sentiment_score'],
    'Actual News Headline': merged_data.loc[X_test.index, 'text'],  # Fetching news headlines from the original data
})

print("\nActual vs. Predicted Values on Test Set with Additional Data:")
print(actual_vs_predicted_with_data)

# Sort the index before plotting
actual_vs_predicted_with_data = actual_vs_predicted_with_data.sort_values('Date')

# Plot a graph of actual vs. predicted closing prices
plt.plot(actual_vs_predicted_with_data['Date'], actual_vs_predicted_with_data['Actual Closing Price'], label='Actual Closing Price', marker='o')
plt.plot(actual_vs_predicted_with_data['Date'], actual_vs_predicted_with_data['Predicted Closing Price'], label='Predicted Closing Price', marker='x')
plt.legend()
plt.xlabel('Date (Test Set)')
plt.ylabel('Closing price')
plt.title('Actual vs. Predicted Closing Prices on Test Set')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()

# Create a larger and wider figure
plt.figure(figsize=(15, 8))

# Plot a graph of actual closing prices for training and test set
plt.plot(merged_data.loc[X_train.index, 'date'], y_train, label='Actual Training Closing Price', marker='o')
plt.plot(actual_vs_predicted_with_data['Date'], actual_vs_predicted_with_data['Actual Closing Price'], label='Actual Test Closing Price', marker='o')

# Add predictions for the test set to the same graph
plt.scatter(actual_vs_predicted_with_data['Date'], actual_vs_predicted_with_data['Predicted Closing Price'], label='Predicted Closing Price', marker='x', color='red')
plt.legend()

plt.xlabel('Date')
plt.ylabel('Closing price')
plt.title('Actual Closing Prices (Training + Test) with Predictions on Test Set')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

plt.show()

# Create a DataFrame with actual and predicted values, sentiment scores, news headlines, and dates for the test set
result_df_test = pd.DataFrame({
    'Date': actual_vs_predicted_with_data['Date'],
    'Actual Closing Price': y_test,
    'Predicted Closing Price': y_pred_test,
    'Sentiment Vader Score': merged_data.loc[X_test.index, 'vader_sentiment_score'],
    'Sentiment TextBlob Score': merged_data.loc[X_test.index, 'textblob_sentiment_score'],
    'News Headline': merged_data.loc[X_test.index, 'text']
})

# Sort the DataFrame by date
result_df_test = result_df_test.sort_values('Date')

# Create a DataFrame with actual values, sentiment scores, news headlines, and dates for the training set
result_df_train = pd.DataFrame({
    'Date': merged_data.loc[X_train.index, 'date'],
    'Actual Closing Price': y_train,
    'Sentiment Vader Score': merged_data.loc[X_train.index, 'vader_sentiment_score'],
    'Sentiment TextBlob Score': merged_data.loc[X_train.index, 'textblob_sentiment_score'],
    'News Headline': merged_data.loc[X_train.index, 'text']
})

# Sort the DataFrame by date
result_df_train = result_df_train.sort_values('Date')

# Combine the training and test dataframes
result_df_combined = pd.concat([result_df_train, result_df_test], axis=0)

# Save the combined DataFrame to an Excel file
result_df_combined.to_excel('result_data_combined.xlsx', index=False)

# Display the combined DataFrame
print("\nCombined Actual vs. Predicted Values with Additional Data:")
print(result_df_combined)
