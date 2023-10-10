# Project Overview: 

The project involves developing two essential models for an American retailer operating ten stores across three different states: California (CA), Texas (TX), and Wisconsin (WI). The retailer sells items from three main categories: hobbies, foods, and household. The objective of this project is to build two predictive models to enhance business operations:

1.	Predictive Sales Revenue Model:

Objective: To accurately predict the sales revenue for a specific item in a particular store on a given date.

Significance: This model is crucial for optimizing inventory management, pricing strategies, and overall store performance. It enables the retailer to make data-driven decisions, reduce wastage, and ensure the availability of popular items.

2.	Forecasting Model:

Objective: To forecast the total sales revenue across all stores and items for the next seven days using time-series analysis.

Significance: This model aids demand forecasting, inventory planning, and budgeting. It allows the retailer to anticipate sales trends, allocate resources efficiently, and respond proactively to market changes.
# Interactive Tableau Visualization
A valuable addition to this repository is the inclusion of an interactive Tableau visualization. For detailed insights, please navigate to the visualization file within the src folder and access the dedicated module visualization file.

# Experiment 1 (predicting)

Due to the importance of specific categorical features, namely item_id and store_id, in our dataset, we strategically chose to utilize the LightGBM model. LightGBM is particularly well-suited for handling categorical variables efficiently and effectively. Its gradient boosting framework, designed to handle categorical data, provides speed and predictive accuracy advantages.
This decision enables us to capitalize on the strengths of LightGBM, ensuring that we can leverage the categorical information in our dataset to improve the model's overall performance and predictive power. By doing so, we aim to obtain more accurate and insightful results considering the nuances of item and store characteristics.

# Experiment 2 (forcasting)
For this experiment, as we require time series models, we conducted runs of ARIMA and SARIMA models with various hyperparameter configurations.

# Deployement

In our Git repository, we have established a "FastAPI" directory. Within this directory, we have included essential files like "requirements.txt," "Dockerfile," and "Docker-compose.yml." 

The "main.py" file plays a pivotal role in our application's heart. Here, we have loaded our machine-learning models and meticulously designed and defined various endpoints. These endpoints serve as the interface through which our application interacts with users and processes incoming requests. In this file, the magic happens as our models are put to work, making predictions and handling data as needed.

Once the Docker image was built, we tested its functionality locally using localhost. With a successful deployment, our application is now live and accessible on Heroku.

To ensure your application functions effectively in the real world, it's crucial to proactively monitor its performance and be ready to scale it when required. Equally vital is addressing security, guarding your application against prevalent vulnerabilities.


