# Table of Contents
- [Project Title](#project-title)
- [Project Objectives](#project-objectives)
- [Questions](#project-questions)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)
- [Results and Findings](#results-findings)
    - [Missing Values](#missing-values)  
    - [Descriptive Statistics](#descriptive-statistics)  
    - [Outliers](#outlier)  
    - [Feature Engineering](#feature-engineering)  
    - [Questions Discussion](#questions-discussion)    
    - [Correlation Matrix](#correlation-matrix)  
- [Model](#model)
    - [Confusion Matrix](#confusion-matrix)  
    - [ROC Curve](#roc-curve)    
    - [Feature Importance](#feature-importance)  
- [Future Works](#future-works)

# [PREDICTING AIRLINE DELAY](#project-title)
This analysis aims to analyse a Kaggle dataset containing historical flight data from 18 airline operators in the United States of America. The goal is to develop a predictive model that anticipates airline delays, enhancing travel planning and operational efficiency. By leveraging historical data, this model will empower travellers to make informed decisions, help airlines optimise resource allocation, and mitigate financial implications. Proactive communication and improved customer experiences are envisioned, contributing to enhanced safety, compliance, and overall satisfaction within the aviation industry.

## [Project Objectives](#project-objectives)
- Analyze historical flight data from 18 airline operators.
- Develop a predictive model to anticipate airline delays.
- Enhance travel planning and operational efficiency.
- Empower travellers and airlines with data-driven insights.

## [Questions](#project-questions)
1. Which airline experiences the most delays?
2. Which routes experience the most delays?
3. Which days of the week experience the most delays?
4. Which days of the week do airlines experience the most delays? 
5. Which routes experience the most delays during the days of the week?


# [Features](#features)
The analysis begins by importing the necessary libraries and conducting exploratory data analysis to gain insightful information regarding the dataset. Following this, supervised machine learning models, namely Random Forest and K-Nearest Neighbor, are applied. The goal is to give readers insights into the factors contributing to flight delays.

# [Technologies](#technologies)
The project utilizes the following technologies and Python libraries:

- Python: The core programming language used for the project.
- Pandas: A powerful data manipulation and analysis library.
- NumPy: A fundamental package for numerical operations in Python.
- Seaborn: A data visualization library based on Matplotlib, providing informative statistical graphics.
- Matplotlib: A comprehensive plotting library for creating static, animated, and interactive visualizations in Python.
- Scikit-learn (sklearn):
    - LabelEncoder: Used for encoding categorical features to numeric values.
    - train_test_split: Utilized for splitting the dataset into training and testing sets.
    - GridSearchCV: Employed for hyperparameter tuning using cross-validated grid search.
    - RandomForestClassifier: A machine learning algorithm used for classification tasks.
    - roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix: Functions for evaluating machine learning model performance.
    - Joblib: Utilized for parallel computing and efficient caching, mainly using parallel_backend and Memory.

These technologies are fundamental in performing data analysis, visualization, preprocessing, modelling, and evaluation within the project.

# [Installation](#installation)
The installation is straightforward and will require you only to run the command below if you decide to run it within a virtual environment (if you do not wish to overwrite existing installations) or in the global environment of your computer's system.
 
 `pip install -r requirements.txt`

This command will read the **requirements.txt** file and install all the listed dependencies, including joblib, matplotlib, numpy, pandas, seaborn, and sklearn since the time module is a fundamental part of Python. The installation will be done using the most recent version of the libraries at the time of installation since versions are not included in the **requirement.txt**.

# [Usage](#usage)
The fastest way to get this project running would be to install [Anacoda](https://www.anaconda.com/). After the installation, you open the Jupyter Notebook and open the folder containing the .ipynb file. Download the [dataset from Kaggle](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay) and place it in the same folder as the .ipynb file.

There is an option to install each library in the requirements.txt file individually by typing **pip install** followed by the library name as shown below. 

`pip install pandas`

# [Folder Structure](#folder-structure)
- airport/
  - airline_delay.ipynb   
  - visualizations/
  - README.md    
  - CODE_OF_CONDUCT.md    
  - requirements.t

# [Contributing](#contributing)
We welcome contributions to improve this project! To contribute, follow these steps:
1. Fork this repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Make and commit your changes: `git commit -m 'Add feature'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Submit a pull request.

## Pull Request Guidelines
- Ensure your code follows the project's style and conventions.
- Explain the changes you've made and briefly overview the problem you're solving.
- Ensure your commits have clear messages.

## Code of Conduct
We have a [Code of Conduct](md/CODE_OF_CONDUCT.md). Please follow it in all your interactions with the project.

## Reporting Issues
If you encounter any issues with the project or would like to suggest enhancements, please open an issue on GitHub or contact me via my social media handles, Twitter (X): @chidiesobe or Instagram @ceesobe. The case will be reviewed and addressed accordingly.

## Contact
If you have any questions, need further assistance, or are just passing by, please contact me via my social media handles, Twitter (X): @chidiesobe or Instagram @ceesobe.

## [License](#license)
This project is licensed under the [MIT License](https://opensource.org/license/mit/).

# [Results and Findings](#results-findings)
This project involved the analysis of the Kaggle dataset earlier referenced in the usage section. We start by getting the general structure of the dataset. The dataset has 539383 rows and 9 columns, comprising 18 airline operators and 293 airports. The columns' names are listed below:

- id: numerical numbering of each row
- Airline: the operating companies operated in and out of each airport.
- Flight: flight number.
- AirportFrom: airport of departure.
- AirportTo: airport of arrival.
- DayOfWeek: numerical representation of days of the week.
- Time: departure time measured in minutes from midnight.	
- Length: duration of the flight in minutes.
- Delay: 0 for not delayed and 1 for delayed.

**NOTE:** The airports and operators are abbreviated, but the full meaning of both columns can be found in the [Abbreviation File](md/ABBREVIATION.md).

## [Missing Values](#missing-values)
A check for missing values returned zero, making the dataset relatively healthy with no missing values in each column or row. While we had no missing values we checked the numerical columns for their minimum and maximum values, and we found the following minimum and maximum values as outlined below: 

- Maximum values:
    - Flight       7814
    - DayOfWeek       7
    - Time         1439
    - Length        655

- Minimum values:
    - Flight        1
    - DayOfWeek     1
    - Time         10
    - Length        0

Observing the minium values showd that the length column had a value of zero (0), but with departure times, meaning the flight operated with no duration of flights. So we dropped all four rows in the dataset with the zero value along side the id column, leaving us with a dataset of 539379 rows and 8 columns.

## [Descriptive Statistics](#descriptive-statistics)
A further look into the dataset revealed the number of times individual airlines operated and the number of times airports were operated from and operated, as shown below: 

- **AIRLINES**
    - WN    94097
    - DL    60940
    - OO    50254
    - AA    45656
    - MQ    36605
- **AIRPORT FROM**
    - ATL    34449
    - ORD    24822
    - DFW    22154
    - DEN    19842
    - LAX    16657
- **AIRPORT TO**
    - ATL    34440
    - ORD    24871
    - DFW    22153
    - DEN    19846
    - LAX    16656

While the information on the operating schedule is very relevant, there was a need to show each airline's respective records of departure and arrival airport while offering the total number of flights, split into delayed and not delayed. This showed a broad picture concerning how each airline contributes to the summary statistics of the dataset.

| Airline | AirportFrom | AirportTo | Delay | Not Delayed |  Total Flights | 
|---------|-------------|-----------|-------|-------------|-----------------
| 9E      | ABE         | DTW       | 45    | 40          | 85             |
|         | ABR         | MSP       | 1     | 1           | 2              |
|         | ALB         | ATL       | 32    | 9           | 41             |
|         |             | DTW       | 51    | 39          | 90             |
|         |             | JFK       | 28    | 3           | 31             |

**Note:** In the Jupyter Notebook file, some cells have codes with .to_csv that can be uncommented to allow the generated dataframe to be exported to a .csv file if there is an interest in viewing the data in Excel or using other .csv readers.

## [Outliers](#outlier)
Box plots and Scatter plots were used to give a visual representation of the outliers in the numerical columns of the dataset, as shown below.

![Scatter plot of Time vs Length](/visualisations/outlier-scatterplot.png)
![Box plot of Numerical Columns](/visualisations/outlier-boxplot.png)

Outliers are data points that stand out from most of the dataset, potentially skewing statistical analyses and machine learning models. The Interquartile Range (IQR) method is a robust statistical tool to detect these outliers. This research calculates the IQR for each numerical column, measuring data spread. The process is repeated iteratively to enhance the outlier detection method. The IQR helps set a threshold beyond which data points are flagged as outliers. These outliers are identified by assessing if data points significantly deviate from the first quartile (Q1) and third quartile (Q3), using a 1.5 times the IQR multiplier. Detected outliers are removed from the dataset, ensuring data accuracy and reliability for subsequent analyses.

Finally, the code displays the count of outliers for each numerical column and presents the dataset after the outlier removal process. This iterative approach to outlier detection using the IQR method allows for more comprehensive and adaptive handling of outliers in varying datasets. 

The box plot and scatter plot below show a visual representation of the dataset after removing outliers. 
![Scatter plot of Time vs Length](/visualisations/outlier-scatterplot2.png)
![Box plot of Numerical Columns](/visualisations/outlier-boxplot2.png)


## [Feature Engineering](#feature-engineering)
Feature engineering was carried out to get better descriptive information about each row of the dataset and also aid in applying machine learning models to predict the likelihood of a flight being delayed. First, two additional columns where created: **FlightDensity**, which represents the number of flights each airline operated and **RouteDensity**, which means the total number of flights each airline carried out on each route (AirportFrom to AirportTo).

Additional feature engineering carried out includes label encoding, a practice of converting categorical variables into numerical representation, making the dataset 502607 rows and 10 columns.


## [Questions Discussion](#questions-discussion)

1. Which airline experiences the most delays?
![Airline with the most delays](/visualisations/most-delay.png)

The bar chart above shows that WN (Southwest Airlines) experienced the most delays while HA (Hawaiian Airlines) experienced the most minor delays.

2. Which routes experience the most delays?
![Routes with the most delays](/visualisations/20-most-delayed-routes.png)

The bar chart shows the 20 most delayed routes, with LAX (Los Angeles International Airport - California) - SFO (San Francisco International Airport - California) experiencing the most delays. 

3. Which days of the week experience the most delays?
![Weeks with the most delays](/visualisations/delay-by-weeks.png)

From the line chart of the weekly delays experienced, we can see that Wednesday and Thursday represent the peak period when passengers would experience the most delays, with Saturday being the least.


4. Which days of the week do airlines experience the most delays? 
Additionally, we narrowed down the delays to determine which days of the week individual airlines experience the most delays. The heatmap shows that WN (Southwest Airlines) experienced the most delays on Wednesday in support of the line chart above. WN generally delays the most across the entire week. With the close, the colour is yellow, representing delays and blue, representing no delay, as shown below. 

![Weeks with the most delays](/visualisations/airline-by-weeks.png)

5. Which routes experience the most delays during the days of the week?
The heat map shows that LAX to SFO experiences significant delays on Mondays and Wednesdays. Other routes with significant delays include ORD to LGA on Wednesday. 

![Weeks with the most delays](/visualisations/top-20-routes.png)


The additional table below shows the top 20 most delayed routes and the number of flights each airline has operating on those routes, with 9E having no flights on the two routes and one of the airlines with moderate delays between flights. There is a lot of information in the table, which is added as an additional reference rather than a point of consideration.


| Route       | 9E  | AA    | AS  | B6  | CO  | DL   | EV  | F9  | FL   | HA  | MQ   | OH  | OO   | UA   | US   | WN  | XE  | YV   |
|-------------|-----|-------|-----|-----|-----|------|-----|-----|------|-----|------|-----|------|------|------|------|-----|-----|
| LAX to SFO  | 0.0 | 185.0 | 0.0 | 0.0 | 0.0 | 62.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0  | 0.0 | 73.0 | 441.0| 0.0  | 318.0| 0.0 | 0.0 |
| SFO to LAX  | 0.0 | 185.0 | 0.0 | 0.0 | 0.0 | 64.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0  | 0.0 | 71.0 | 440.0| 0.0  | 317.0| 0.0 | 0.0 |
| LAX to LAS  | 0.0 | 123.0 | 0.0 | 0.0 | 0.0 | 12.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0  | 0.0 | 263.0| 56.0 | 63.0 | 337.0| 0.0 | 74.0|
| DAL to HOU  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 | 0.0  | 0.0 | 0.0  | 0.0 | 0.0  | 0.0  | 0.0  | 701.0| 0.0 | 0.0 |
| HOU to DAL  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 | 0.0  | 0.0 | 0.0  | 0.0 | 0.0  | 0.0  | 0.0  | 698.0| 0.0 | 0.0 |
| LAS to LAX  | 0.0 | 123.0 | 0.0 | 0.0 | 0.0 | 12.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0  | 0.0 | 261.0| 57.0 | 63.0 | 338.0| 0.0 | 74.0|
| ATL to LGA  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 474.0| 0.0 | 0.0 | 237.0| 0.0 | 204.0| 0.0 | 0.0  | 0.0  | 0.0  | 0.0  | 0.0 | 0.0 |
| LAX to SAN  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 | 0.0  | 0.0 | 362.0| 0.0 | 573.0| 0.0  | 0.0  | 0.0  | 0.0 | 0.0 |
| DEN to SLC  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 2.0  | 0.0 | 59.0| 0.0  | 0.0 | 0.0  | 0.0 | 398.0| 20.0 | 0.0  | 168.0| 0.0 | 0.0 |
| ATL to MCO  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 490.0| 0.0 | 0.0 | 288.0| 0.0 | 0.0  | 0.0 | 0.0  | 0.0  | 0.0  | 0.0  | 0.0 | 0.0 |
| ORD to LGA  | 0.0 | 432.0 | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 | 0.0  | 0.0 | 0.0  | 0.0 | 0.0  | 410.0| 0.0  | 0.0  | 0.0 | 0.0 |
| DEN to LAX  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 |176.0| 0.0  | 0.0 | 123.0| 0.0 | 0.0  | 265.0| 0.0  | 212.0| 0.0 | 0.0 |
| LAX to DEN  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 |177.0| 0.0  | 0.0 | 123.0| 0.0 | 0.0  | 254.0| 0.0  | 214.0| 0.0 | 0.0 |
| LAS to PHX  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 | 0.0  | 0.0 | 0.0  | 0.0 | 0.0  | 0.0  | 312.0| 401.0| 0.0 | 31.0|
| SAN to LAX  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 | 0.0  | 0.0 | 362.0| 0.0 | 573.0| 0.0  | 0.0  | 0.0  | 0.0 | 0.0 |
| LGA to ATL  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 475.0| 0.0 | 0.0 | 237.0| 0.0 | 204.0| 0.0 | 0.0  | 0.0  | 0.0  | 0.0  | 0.0 | 0.0 |
| PHX to LAS  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 | 0.0  | 0.0 | 0.0  | 0.0 | 0.0  | 0.0  | 311.0| 406.0| 0.0 | 30.0|
| SLC to DEN  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 2.0  | 0.0 | 59.0| 0.0  | 0.0 | 0.0  | 0.0 | 404.0| 45.0 | 0.0  | 171.0| 0.0 | 0.0 |
| ORD to DFW  | 0.0 | 451.0 | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 | 0.0  | 0.0 | 0.0  | 0.0 | 31.0 | 83.0 | 0.0  | 0.0  | 0.0 | 0.0 |
| ATL to FLL  | 0.0 | 0.0   | 0.0 | 0.0 | 0.0 | 414.0| 0.0 | 0.0 | 249.0| 0.0 | 0.0  | 0.0 | 0.0  | 0.0  | 0.0  | 0.0  | 0.0 | 0.0 |
| ........... | ... | ...   | ... | ... | ... | ...  | ... | ... | ...  | ... | ...  | ... | ...  | ...  | ...  | ...  | ... | ... |


# [Correlation Matrix](#correlation-matrix)
![Weeks with the most delays](/visualisations/correlation-matix.png)

The correlation matrix provides insights into the linear relationships between pairs of variables. Here's an analysis of the correlation matrix you provided:

- The magnitude of Correlation: The correlation values range from -1.00 to 1.00. A value closer to 1.00 or -1.00 indicates a stronger linear relationship, while values closer to 0 suggest a weaker relationship.

- Self-Correlation (Diagonal): The self-correlation of a variable is always 1.00 since it's a correlation of a variable with itself.

- Correlation Between Variables: FlightDensity has a strong positive correlation (0.43) with Airline, indicating that the corresponding airline also tends to have higher values as flight density increases. Length has a notable negative correlation (-0.22) with Airline, suggesting that longer flights are associated with lower values for the airline.
FlightDensity has a moderate positive correlation (0.21) with Delay, indicating that higher flight density might be associated with higher delays. Length has a moderate negative correlation (-0.15) with RouteDensity, suggesting that longer flights might be associated with lower route density.

Positive correlations suggest that as one variable increases, the other tends to increase as well (and vice versa for negative correlations).

# [Model](#model)

The modelling phase started with randomly sampling 10,000 records from the feature engineered dataset. Some of the reasons behind the sampling include: 

1. **Data Size Reduction**:
   - Working with a large dataset can be computationally intensive and time-consuming, especially during model training and evaluation. By sampling a smaller subset, we can significantly reduce the computational load while retaining meaningful insights.

2. **Representative Sample**:
   - The process aims to obtain a representative subset that maintains the essential characteristics and patterns present in the original dataset. Random sampling ensures a fair representation of the dataset's diversity.

3. **Resource Efficiency**:
   - Training machine learning models on a smaller dataset is faster, making it easier to experiment with various algorithms, features, and hyperparameters. This expedites the model development and iteration process.


While the visualisation was carried out using the original dataset, sampling 10,000 records is a strategic step to reduce computational load, improve efficiency, and maintain essential insights while building and evaluating machine learning models. It allows for effective exploration of the data and facilitates efficient model development. The sampled dataset has **55.36%** representing not delayed data while **44.64%** representing the delayed data.

![Weeks with the most delays](/visualisations/distribution.png)

In this project phase, the model underwent evaluation and hyperparameter tuning. One hundred eight hyperparameter combinations were tested, employing techniques like grid search. The resulting best hyperparameters for the model were identified as **'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, and 'n_estimators': 50**. Moving on to evaluating the model's performance, an accuracy of approximately **63.77%** was achieved on the evaluation dataset. Precision, recall, and F1-score metrics were computed for each class, providing a detailed understanding of the model's predictive capabilities for delays (class '1') and non-delays (class '0'). 
Notably, precision and recall were higher for non-delays, indicating better model performance in identifying non-delayed flights. The evaluation also factored in the time to run the model, which amounted to around 88.70 seconds. This analysis is pivotal in comprehending the model's efficiency, determining optimal hyperparameters, and gauging its effectiveness in predicting flight delays.

While we randomly sampled 10,000 rows for the model building and evaluation stage, the model is created to accommodate the entire dataset size, meaning the sampling section can be commented out, and the model would run just fine. To achieve this, the Memory object from the joblib library handles caching and memory management, thereby addressing potential memory leaks. Memory caching helps store the results of expensive function calls and reuse them when the same inputs occur again, saving computation time and resources.

The Memory object is initialized with a specified location for caching (location='./cachedir') and verbosity level (verbose=0). The **@memory.cache** decorator is then applied to the fit_random_forest function. This decorator allows the process to be cached, so if the same procedure is called with the same parameters in the future, the previously computed result can be reused from the cache instead of recomputing.

By utilizing this caching mechanism, the code ensures that the computationally intensive **fit_random_forest** function is only executed when necessary, and the results are stored and retrieved efficiently from the cache when the same process is called with the same inputs. This approach helps manage memory effectively and mitigate potential memory leaks that could occur during the execution of the script.

## [Confusion Matrix](#confusion-matrix)
![Correlation Matrix](/visualisations/confusion-matrix.png)

In the confusion matrix:

- True 0 (Predicted 0): There are 1356 instances where the model correctly predicted a delay (class 0), and it was indeed a delay.

- True 1 (Predicted 1): There are 557 instances where the model correctly predicted not a delay (class 1), and it was indeed not a delay.

- False 0 (Predicted 0, Actual 1): There are 807 instances where the model predicted a delay (class 0), but it was not a delay (class 1). These are false positives.

- False 1 (Predicted 1, Actual 0): There are 280 instances where the model predicted not a delay (class 1), but it was a delay (class 0). These are false negatives.

## [ROC Curve](#roc-curve)
![Roc Curve](/visualisations/roc-curve.png)

The ROC curve visually demonstrates the trade-off between true and false positive rates, helping you choose an appropriate classification threshold for your model based on the problem's context and priorities. An ROC area (or AUC - Area Under the Curve) of 0.68 means that the model's ability to distinguish between the two classes (e.g., positive and negative cases in a binary classification problem) is moderate.

## [Feature Importance](#feature-importance)
Feature Importance indicates the contribution of each feature (or variable) towards making predictions. It helps in understanding which features are more influential in the model's decision-making process.

| Feature       | Importance   |
|---------------|--------------|
| Time          | 0.2107       |
| Flight        | 0.1371       |
| FlightDensity | 0.1291       |
| Length        | 0.1130       |
| RouteDensity  | 0.0991       |
| AirportFrom   | 0.0978       |
| AirportTo     | 0.0863       |
| Airline       | 0.0728       |
| DayOfWeek     | 0.0543       |


Time (0.2107): This feature has the highest importance. It suggests that the 'Time' variable significantly influences the model's predictions, making it a crucial factor in determining delays.

**Flight (0.1371):** The 'Flight' feature is also important, though slightly less than 'Time'. It implies that the flight number plays a substantial role in predicting delays.

**FlightDensity (0.1291):** 'FlightDensity' is the third most important feature. A high value here indicates that the density of flights is influential in predicting delays.

**Length (0.1130):** The 'Length' feature is important and contributes significantly to the model's predictions.

**RouteDensity (0.0991):** 'RouteDensity' is also a substantial factor in determining delays.

**AirportFrom (0.0978) and AirportTo (0.0863):** These features, representing the departure and arrival airports, are also crucial in predicting delays. Different airports may have varying patterns of delays.

**Airline (0.0728):** The 'Airline' feature represents the airline operating the flight and is important in predicting delays. Different airlines may have different delay patterns.

**DayOfWeek (0.0543):** The day of the week is also a factor in predicting delays, although it has a relatively lower importance compared to other features.

Understanding feature importance is essential for optimizing the model, identifying key factors affecting delays, and potentially improving airline operations to reduce delays.

The feature importances can be visually shown using the bar chart below: 

![Feature Importances](/visualisations/feature-imp.png)

# [Future Works](#future-works)

Here are potential enhancements and plans for this project:

**Real-time Prediction:** Integrate the predictive model into a real-time application to provide travellers with live predictions of flight delays based on current conditions.

**Enhanced Features:** Incorporate additional relevant features such as weather data, airport traffic conditions, and historical airline performance data to enhance predictive accuracy.

**Ensemble Modeling:** Explore ensemble modelling techniques like stacking or bagging to improve the predictive performance of the model further.

**Interactive Dashboard:** Develop an interactive dashboard that allows users to explore the data, visualize insights, and obtain delay predictions for specific flights and routes.

**Feature Engineering:** Explore additional feature engineering techniques to create new informative features that might enhance model predictions.

**Hyperparameter Tuning:** Continue experimenting with hyperparameters to find the best configuration that optimizes the model's performance.

**Deployment and Scalability:** Optimize the model for deployment, ensuring scalability and efficiency in handling many user prediction requests.

The future works aim to enhance the predictive model, provide more actionable insights to travellers, and facilitate better decision-making for travellers and airlines. Through advancements in technology and analytics, we aspire to contribute to a more efficient and pleasant air travel experience for all stakeholders involved.