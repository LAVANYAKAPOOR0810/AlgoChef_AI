# ALGOCHEF : ALGORITHM FOR AMAZING MEALS


## Project Overview :
AlgoChef is a machine learning and NLP-based web application that recommends recipes to users based on:
* Ingredients available with the user
* Nutritional preferences
* Similarity between recipes
* User-selected food inputs

--------------------

## The project demonstrates practical implementation of:
* KNN Algorithm
* TF-IDF Vectorization
* Euclidean Distance
* Feature Engineering
* Model Serialization
* Full-stack Web Integration

--------------------

## 💻 Tech Stack
### Frontend
* HTML
* CSS
* JavaScript
* Bootstrap
* React (Shopzy Integration)
  
### Backend
* Flask
* Python

### AI / ML Libraries :
* scikit-learn	Machine learning
* pandas	Dataset handling
* numpy	Numerical operations
* scipy	Sparse matrix operations
* pickle	Model saving/loading


--------------------

## Artificial Intelligence Concepts Used :
1. Machine Learning (ML)
* The system learns relationships between ingredients and nutrition values
* Recipes are converted into numerical vectors
* KNN algorithm finds recipes that are most similar
* The trained model predicts nearest recipes based on similarity

2. Natural Language Processing (NLP)
NLP is used because ingredients are textual data.
The system cannot directly understand text, so NLP converts ingredient text into machine-readable vectors.

* This is done using -
  -TF-IDF Vectorization

* Core Algorithm Used -
 - K-Nearest Neighbors (KNN) 

Step 1: Dataset Loading
 - recipe_df = pd.read_csv("recipe_final1.csv")

Step 2: Ingredient Text Processing
 - vectorizer = TfidfVectorizer()
 - X_ingredients = vectorizer.fit_transform(recipe_df["ingredients_list"])

Step 3: Numerical Feature Scaling
 - scaler = StandardScaler()

Step 4: Sparse Matrix Conversion
 - X_numerical_sparse = csr_matrix(X_numerical)

Step 5: Combining Features
  -X_combined = hstack([X_ingredients, X_numerical_sparse])

* Euclidean Distance
 - KNN calculates similarity using Euclidean Distance.

* Smaller distance means:
  -Recipes are more similar
* Larger distance means:
  -Recipes are different

* Model Training :
 - knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
 - knn.fit(X_combined)

* Model Saving :
 - pickle.dump(knn, open("recipe_model.pkl", "wb"))

* Saved files :
 - recipe_model.pkl	Trained KNN model
 - vectorizer.pkl	TF-IDF vocabulary
 - scaler.pkl	Feature scaling values


--------------------

## Recommendation Process :
When user enters ingredients:
1. Input ingredients converted into TF-IDF vector
2. Nutritional values scaled
3. Features combined
4. Euclidean distance calculated
5. Nearest recipes selected
6. Top recommendations displayed

Project Architecture :

- User Input
-     ↓
- Frontend Interface
-     ↓
- Flask Backend
-     ↓
- Input Preprocessing
-     ↓
- TF-IDF Vectorization
-     ↓
- Feature Scaling
-     ↓
- Feature Combination
-     ↓
- KNN Similarity Search
-     ↓
- Recipe Recommendations
-     ↓
- Display Results



--------------------

## Project Files Description :
* app.py	Main Flask application
* recipe_model.py	Model training script
* recipe_model.pkl	Trained KNN model
* vectorizer.pkl	TF-IDF vectorizer
* scaler.pkl	Feature scaler
* recipe_final1.csv	Dataset
* templates/	HTML pages
* static/	CSS/JS files


--------------------

## Future Improvements :
* Deep Learning-based recommendations
* User login and personalization
* Collaborative filtering
* Ingredient image recognition
* Voice-based recipe search
* Real-time grocery pricing


--------------------

## To Run :
1. cd Recipe_Recommendation_System
2. Install Required Libraries
   pip install -r requirements.txt
3. Train the Model
   python train_model.py
This generates:
  * recipe_model.pkl
  * vectorizer.pkl
  * scaler.pkl
4. Run the Flask Server
   python app.py



