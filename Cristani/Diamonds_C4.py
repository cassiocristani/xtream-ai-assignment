import pandas as pd
<<<<<<< Updated upstream
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
=======
from flask import Flask, jsonify, request, render_template, redirect, url_for 
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from joblib import dump
>>>>>>> Stashed changes
import seaborn as sns
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import os
import time
<<<<<<< Updated upstream

app = Flask("__Diamonds__")
teammates = []

@app.route('/plot', methods=['GET'])
def plot_model(data_path, model, y_test, y_pred, margin, evaluated_sample_index):
     # Create a DataFrame for visualization
    df_visualization = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': y_pred})

    # Calculate the differences between predicted and actual prices
    df_visualization['Price Difference'] = df_visualization['Predicted Prices'] - df_visualization['Actual Prices']

    # Calculate the absolute differences
    df_visualization['Absolute Difference'] = abs(df_visualization['Price Difference'])

    # Set a threshold for underestimation (adjust as needed)
    underestimate_threshold = 0

    # Identify underestimated samples
    df_visualization['Underestimated'] = df_visualization['Price Difference'] < underestimate_threshold

    # Identify overestimated samples
    df_visualization['Overestimated'] = df_visualization['Price Difference'] > underestimate_threshold

    # Filter out samples within a margin 
    margin = 0.10 
    avg_abs_diff_overestimated = df_visualization.loc[df_visualization['Overestimated'], 'Absolute Difference'].mean()
    avg_abs_diff_underestimated = df_visualization.loc[df_visualization['Underestimated'], 'Absolute Difference'].mean()

    df_filtered_overestimated = df_visualization.loc[df_visualization['Overestimated'] & (df_visualization['Absolute Difference'] > (1 + margin) * avg_abs_diff_overestimated)]
    df_filtered_underestimated = df_visualization.loc[df_visualization['Underestimated'] & (df_visualization['Absolute Difference'] > (1 + margin) * avg_abs_diff_underestimated)]

    # Count the number of original samples
    total_samples = len(df_visualization)

    # Count the number of samples after filtering
    total_samples_filtered = len(df_filtered_overestimated) + len(df_filtered_underestimated)
    total_overestimated_filtered = len(df_filtered_overestimated)
    total_underestimated_filtered = len(df_filtered_underestimated)

    # Calculate average and standard deviation for overestimated and underestimated samples after filtering
    avg_overestimated_filtered = df_filtered_overestimated['Absolute Difference'].mean()
    std_overestimated_filtered = df_filtered_overestimated['Absolute Difference'].std()

    avg_underestimated_filtered = df_filtered_underestimated['Absolute Difference'].mean()
    std_underestimated_filtered = df_filtered_underestimated['Absolute Difference'].std()

    # Print the number of original samples
    print(f'Total original samples: {total_samples}')

    # Print the number and percentage of samples after filtering
    print(f'Total samples after filtering ( {100*margin} %): {total_samples_filtered} ({total_samples_filtered / total_samples * 100:.2f}%)')
    print(f'Total overestimated samples after filtering (> {underestimate_threshold} and margin {margin}): {total_overestimated_filtered} '
          f'({total_overestimated_filtered / total_samples * 100:.2f}%)')
    print(f'Total underestimated samples after filtering (< {underestimate_threshold} and margin {margin}): {total_underestimated_filtered} '
          f'({total_underestimated_filtered / total_samples * 100:.2f}%)')

    # Print average and standard deviation for overestimated and underestimated samples after filtering
    print(f'Average absolute difference for overestimated samples after filtering: {avg_overestimated_filtered:.2f}')
    print(f'Standard deviation for overestimated samples after filtering: {std_overestimated_filtered:.2f}')

    print(f'Average absolute difference for underestimated samples after filtering: {avg_underestimated_filtered:.2f}')
    print(f'Standard deviation for underestimated samples after filtering: {std_underestimated_filtered:.2f}')

    # Plot the differences with actual prices on the x-axis after filtering
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Actual Prices', y='Price Difference', hue='Predicted Prices', data=df_filtered_overestimated)
    sns.scatterplot(x='Actual Prices', y='Price Difference', hue='Predicted Prices', data=df_filtered_underestimated)
    plt.axhline(y=0, color='black', linestyle='--', label='Zero Difference')
    plt.axhline(y=avg_overestimated_filtered, color='red', linestyle='--', label=f'Avg Overestimated: {avg_overestimated_filtered:.2f}')
    plt.axhline(y=-avg_underestimated_filtered, color='blue', linestyle='--', label=f'Avg Underestimated: {avg_underestimated_filtered:.2f}')

    plt.xlabel('Actual Prices')
    plt.ylabel('Price Difference (Predicted - Actual)')
    plt.title(f'Linear Regression: Differences between Predicted and Actual Prices (Filtered, Margin={margin * 100}%)')
    plt.legend(title='Predicted Prices')
    plt.show()

=======
from werkzeug.utils import secure_filename

#curl -X POST -F "file=C:\\Users\\cassiorodrigo.crisfa\\Documents\\GitHub\\xtream-ai-assignment\\Cristani\\buffer_c4\\diamonds_volsym.csv" http://localhost:5000/add_files
#curl -X POST -F "file=C:\\Users\\cassiorodrigo.crisfa\\Documents\\GitHub\\xtream-ai-assignment\\Cristani\\buffer_c4\\diamonds_volsym.csv" http://localhost:5000/add_files

app = Flask("Diamonds_APP")
app.config['SECRET_KEY'] = 'precious'
data_directory = "C:\\Users\\cassiorodrigo.crisfa\\Documents\\GitHub\\xtream-ai-assignment\\Cristani\\buffer_c4\\"
model_path = "C:\\Users\\cassiorodrigo.crisfa\\Documents\\GitHub\\xtream-ai-assignment\\Cristani\\buffer_c4\\model.joblib"

def save_model(model, model_path):
    try:
        dump(model, model_path)
        print(f"Model saved successfully to {model_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")
        
>>>>>>> Stashed changes
def converting_to_numeral(csv_path): 
    #Presupose that there is a backup of the file. OR should create a backup function to do that
    
    # Read the original CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Mapping dictionaries for 'cut', 'clarity', and 'color'
    cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    clarity_mapping = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
    color_mapping = {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7, 'K': 8, 'L': 9, 'M': 10,
                     'N': 11, 'O': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'U': 18, 'V': 19, 'W': 20,
                     'X': 21, 'Y': 22, 'Z': 23}

    # Convert 'cut', 'clarity', and 'color' to integers
    df['cut'] = df['cut'].map(cut_mapping)
    df['clarity'] = df['clarity'].map(clarity_mapping)
    df['color'] = df['color'].map(color_mapping)


    # Calculate 'z_depth' and 'table_width' columns
    df['z_depth'] = df['depth'] * df['z'] * 100
    df['table_width'] = df['table'] * df['x'] * 100

<<<<<<< Updated upstream
    # Drop rows with NaN values (if any)
    df.dropna(inplace=True)


    df.to_csv(csv_path, index=False)

=======
>>>>>>> Stashed changes
    # Create 'color_classification' column 
    df['color_classification'] = pd.Series(dtype='int')
    df.loc[(df['color'] >= 1) & (df['color'] <= 3), 'color_classification'] = 1 # 'Colorless'
    df.loc[(df['color'] >= 4) & (df['color'] <= 7), 'color_classification'] = 2 #'Near Colorless'
    df.loc[(df['color'] >= 8) & (df['color'] <= 10), 'color_classification'] = 3 #'Faint Yellow'
    df.loc[(df['color'] >= 11) & (df['color'] <= 15), 'color_classification'] = 4 #'Very Light Yellow'
    df.loc[(df['color'] >= 16) & (df['color'] <= 23), 'color_classification'] = 5 #'Light Yellow'
    # Create 'color_classification' column 
    color_classification_mapping = {'Colorless': 1, 'Near Colorless': 2, 'Faint Yellow': 3, 'Very Light Yellow': 4, 'Light Yellow': 5}
    df['color_classification'] = df['color_classification'].map(color_classification_mapping)
<<<<<<< Updated upstream
=======
    # Drop rows with NaN values (if any)
    df.dropna(inplace=True)
    df.to_csv(csv_path, index=False)
>>>>>>> Stashed changes

    return csv_path

def regularize_original_file(dataset_path):
    # List of columns to keep
    columns_regularized = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'price']
    
    # Read the CSV file into a DataFrame
    data = pd.read_csv(dataset_path)
    
    # Drop columns not in the specified list
    columns_to_drop = [col for col in data.columns if col not in columns_regularized]
    data.drop(columns=columns_to_drop, inplace=True)
    
<<<<<<< Updated upstream
=======
    # Drop rows with NaN values (if any)
    data.dropna(inplace=True)

>>>>>>> Stashed changes
    # Save the regularized DataFrame to a new CSV file
    data.to_csv(dataset_path, index=False)
    
    return dataset_path
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
               
def test_reg_file(dataset_path):
    # columns of regularized file
    columns_regularized = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'price']
    
    data = pd.read_csv(dataset_path)
    
    # Check if columns are equal to the regularized list
    if list(data.columns) == columns_regularized:
        return True
    else:
        return False
    
<<<<<<< Updated upstream
@app.route('/explain', methods=['GET'])
def explain_model(model, explainer, X_test, instance_index, feature_names):

    instance = X_test.iloc[instance_index].values
    prediction = model.predict([instance])[0]

    explanation = explainer.explain_instance(instance, model.predict, num_features=len(feature_names))

    # Print explanation
    print("LIME Explanation:")
    print(f"Predicted Price: {prediction}")
    print("Feature Weights:")
    for feature, weight in explanation.as_list():
        print(f"{feature}: {weight}")
    explanation.show_in_notebook()
    
@app.route('/train_buffer', methods=['POST'])
def buffer_feeder(data_directory, backup_directory, model_path, buffer_sleep_time, margin):
=======
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model
    
def split_price(csv_path):
    df = pd.read_csv(csv_path)

    # Extract features (X) and target variable (y)
    X = df.drop(columns=['price']) 
    y = df['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def buffer_feeder(data_directory, model_path, margin=0.1):
>>>>>>> Stashed changes
    while True:
        for dataset_file in os.listdir(data_directory):
            dataset_path = os.path.join(data_directory, dataset_file)
            
<<<<<<< Updated upstream
            dataset_path = converting_to_numeral(dataset_path)
            # Drop columns if not regularized
            if test_reg_file(dataset_path) == False:
                filepath = regularize_original_file(dataset_path)
            
            # Load and preprocess the data
            X_train, X_test, y_train, y_test = split_price(dataset_path)
=======
            converted_path = converting_to_numeral(dataset_path)
            # Drop columns if not regularized
            if test_reg_file(converted_path) == False:
                filepath = regularize_original_file(converted_path)
            else:
                filepath = converted_path
            # Load and preprocess the data
            X_train, X_test, y_train, y_test = split_price(filepath)
            
>>>>>>> Stashed changes

            # Train the model
            model = train_model(X_train, y_train)
            y_pred = model.predict(X_test)
            # Evaluate the model
<<<<<<< Updated upstream
            mse, percentage_within_margin = evaluate_model(model, X_test, y_test, margin)
            print(f'Mean Squared Error: {mse}')
            print(f'Percentage of Samples within {margin * 100}%: {percentage_within_margin:.2f}%')

            os.remove(filepath)
            # Save the trained model
            save_model(model, model_path)
            plot_model(dataset_path, model, y_test, y_pred, margin, evaluated_sample_index=1)
            explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, mode="regression", feature_names=X_train.columns)
            
            while True:
                user_input = input("Enter the number of the samples to explain (or 'x' to exit): ").lower()
                if user_input == 'x':
                    break  # Break out of the loop if the user wants to exit
                try:
                    sample = int(user_input)
                    # Call the explain_model function
                    explain_model(model, explainer, X_test, sample, X_train.columns)
                except ValueError:
                    print("Invalid input. Please enter a valid integer or 'x' to exit.")

            # Ask if the user needs another explanation for a new sample
            another_explanation = input("Do you need explain another sample? (y/n): ").lower()
            if another_explanation != 'y':
                break

            # Move the processed dataset file to another directory or delete it
            processed_dataset_path = os.path.join(backup_directory, 'processed', dataset_file)
            os.rename(dataset_path, processed_dataset_path)
            
            print(f"Waiting for {buffer_sleep_time} seconds before checking for new datasets...")
            for _ in range(buffer_sleep_time):
                time.sleep(1)  # Sleep for 1 second, repeated buffer_sleep_time times
                
                user_input = input("Enter 'x' to exit or press Enter to continue: ").lower()
                if user_input == 'x':
                    break  # Break out of the main loop if the user wants to exit

"""     
@app.route('/explain', methods=['POST'])
def explain_sample():
    data = request.get_json()

    dataset_path = converting_to_numeral(data['csv_path'])
    if not test_reg_file(dataset_path):
        filepath = regularize_original_file(dataset_path)

    X_train, X_test, y_train, y_test = split_price(dataset_path)
    model = train_model(X_train, y_train)

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, mode="regression", feature_names=X_train.columns)
    
    sample_index = data['sample_index']
    instance = X_test.iloc[sample_index].values

    explanation = explainer.explain_instance(instance, model.predict, num_features=len(X_train.columns))
    
    explanation_data = {
        "predicted_price": float(model.predict([instance])[0]),
        "explanation": [{"feature": feature, "weight": weight} for feature, weight in explanation.as_list()]
    }

    return jsonify(explanation_data)
"""

@app.route('/add_teammate', methods=['POST'])
def add_teammate():
    data = request.get_json()

    # Validate if the required fields are present in the request data
    if 'id' not in data or 'name' not in data:
        return jsonify({"error": "Missing required fields. Please provide 'id', 'name."}), 400

    # Create a new teammate
    new_teammate = {
        'id': data['id'],
        'name': data['name'],
    }

    # Add the new teammate to the data store
    teammate.append(new_teammate)

    return jsonify({"message": "Customer added successfully", "customer": new_customer}), 201

@app.route('/delete_teammate/<int:teammate_id>', methods=['DELETE'])
def delete_teammate(customer_id):
    global teammate

    # Find the customer with the given ID
    teammate_to_delete = next((teammate for teammate in teammate if teammate['id'] == teammate_id), None)

    if teammate_to_delete:
        teammate = [teammate for teammate in teammate if teammate['id'] != teammate_id]
        return jsonify({"message": f"Teammate with ID {teammate_id} deleted successfully"}), 200
    else:
        return jsonify({"error": f"Teammate with ID {teammate_id} not found"}), 404


@app.route('/add_files', methods=['PUT'])
def add_file_to_buffer(file_path, buffer_directory):
    # Check if the buffer directory exists, create it if not
    if not os.path.exists(buffer_directory):
        os.makedirs(buffer_directory)

    # Extract the file name from the path
    file_name = os.path.basename(file_path)

    # Create the destination path in the buffer directory
    destination_path = os.path.join(buffer_directory, file_name)

    # Move the file to the buffer directory
    shutil.move(file_path, destination_path)

    print(f"File '{file_name}' added to the buffer directory.")
    
=======
            #mse, percentage_within_margin = evaluate_model(model, X_test, y_test, margin)
            #print(f'Mean Squared Error: {mse}')
            #print(f'Percentage of Samples within {margin * 100}%: {percentage_within_margin:.2f}%')
            #print("filepath", filepath)

            # Save the trained model
            save_model(model, model_path)
            os.remove(filepath)
            
class UploadFileForm(FlaskForm):
    file = FileField("file")
    submit = SubmitField("Upload File")

@app.route("/", methods=["GET", "POST"])
def upload_file():
    form = UploadFileForm()

    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file_path = os.path.join(data_directory, filename)
        file.save(file_path)  # Save the file to the data_directory
        return redirect(url_for('upload_file'))

    return render_template('index.html', form=form)

            
@app.errorhandler(404)
def invalid_route(e):
    return "Invalid Route!"    


@app.route("/add_files", methods=["POST"])
def add_files():
    # Get all files from the request
    files = request.files.getlist("files")

    # Send all files in the directory
    added_files = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(data_directory, filename)
        file.save(file_path)
        added_files.append(filename)

    return jsonify({"message": "Files added to the buffer successfully", "added_files": added_files})

@app.route("/train_model", methods=["POST"])
def train_model_endpoint():
    # Call your buffer_feeder function
    print("passed- UUUF")
    buffer_feeder(data_directory, model_path, margin=0.1)

    return jsonify({"message": "Model trained successfully"})
>>>>>>> Stashed changes

if __name__ == '__main__':
    app.run(debug=True)