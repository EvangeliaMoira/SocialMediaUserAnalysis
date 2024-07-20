# SocialMediaUserAnalysis

This repository contains a comprehensive analysis of social media users, focusing on their interests, demographics, and behavior patterns. The project employs various data analysis techniques, including association rule mining using the Apriori algorithm, clustering with K-means, and visualizations to uncover insights from the dataset.

## Project Overview

- **Data Preprocessing**: Cleaning and transforming data, including handling categorical variables and calculating user ages from birth dates.
- **Association Rule Mining**: Applying the Apriori algorithm to discover frequent itemsets and generate association rules among user interests and names.
- **Clustering Analysis**: Utilizing K-means clustering to segment users based on age and UserID, and determining the optimal number of clusters using the elbow method.
- **Visualization**: Creating histograms and other plots to visualize the distribution of user ages and other demographic information.

## Technologies Used

- Python
- Pandas
- Mlxtend
- Scikit-learn
- Matplotlib
- Seaborn

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SocialMediaUserAnalysis.git

2. Navigate to the project directory:
   cd SocialMediaUserAnalysis

3. Install the required packages:
   pip install -r requirements.txt

4. Place the dataset (SocialMediaUsersDataset.csv) in the project directory.

5. Run the analysis script:
   python analysis.py
