-- Food Delivery Time Predictor --
-------------------------------------------
Wonder how UberEats , Talabat , DoorDash and more food ordering apps predict how long the order will take to reach your doorstep ? This is a machine learning project that predicts food delivery times using neural networks. 

The model considers features such as:

Distance: How far is the restaurant from your house?
Weather: Is it raining or sunny?
Traffic: Rush hour or smooth sailing?
Vehicle Type: Motorcycle ninja or scooter cruiser?
Driver Rating: Is your driver a delivery legend?
Festival Season: Is everyone ordering food today?
Time Factors: Late night munchies or lunch rush?

Pipeline includes: 

Deep Learning Model: TensorFlow/Keras neural network with dropout regularization
Feature Engineering: Distance calculation using Haversine formula, time-based features extraction
Data Preprocessing: Handles missing values, categorical encoding, and feature scaling
Performance Metrics: MAE, RMSE, and R² score evaluation
Scenario Testing: Predicts delivery times for different real-world scenarios

Model Performance

Mean Absolute Error: ~X.XX minutes
Root Mean Square Error: ~X.XX minutes
R² Score: X.XXX

Tech Stack

Python 3.x
TensorFlow/Keras - Deep learning framework
pandas & NumPy - Data manipulation
scikit-learn - Preprocessing and metrics
matplotlib - Data visualization
