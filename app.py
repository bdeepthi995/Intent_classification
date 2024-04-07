from flask import Flask, render_template, request, make_response
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('temp_dataset.csv')  # Change 'intent.csv' to the name of your CSV file
X_train = dataset['Text']
y_train = dataset['Intent']

# Create a CountVectorizer to convert text data to numerical form
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_intent():
    try:
        # Check if a file was uploaded
        if 'csv_file' not in request.files:
            return "No file uploaded"

        # Read the uploaded CSV file
        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            return "Empty file name"

        df = pd.read_csv(csv_file)

        # Vectorize the input text
        input_text_vectorized = vectorizer.transform(df['Text'])

        # Predict the intents
        predicted_intents = model.predict(input_text_vectorized)

        # Add predicted intents to the DataFrame
        df['Predicted Intent'] = predicted_intents

        # Convert DataFrame to CSV
        output_csv = df.to_csv(index=False)

        # Create response with CSV data
        response = make_response(output_csv)
        response.headers['Content-Disposition'] = 'attachment; filename=result.csv'
        response.headers['Content-type'] = 'text/csv'

        return response

    except Exception as e:
        return f"Internal Server Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
