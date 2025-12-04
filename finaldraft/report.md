# Gentrification Classifier
By: Sahith Karra

### Research Question
Recently, there has been a trend of "performative" males who engage in activities such as drinking matcha lattes, reading feminist literature, using wired earbuds, and more. When I traveled to San Francisco and other parts of the Bay, I realized how many people follow this archetype, either ironically or unironically. This led me to another line of thought—how different Bay Area landscapes are. There are old dilapidated neighborhoods, suburban communities, and trendy areas where I could find these "performative" males.

This made me ask the core question of this research experiment: could a given neighborhood's business type be a predictor of rent price increase? Basically, could the influx of trendy and hip businesses such as matcha cafes, yoga studios, KBBQ restaurants, etc., be a sign that the neighborhood is gentrified?

> Gentrification: The process of transforming a lower-value urban neighborhood into a higher-value one through the influx of wealthier residents, typically characterized by the increase in cost of living.

### Data Sources
Rent Prices  
Rather than find a shallow "gentrification" dataset with ambiguous class labels, I crafted my own gentrification labels using historic rent price data. I relied on Zillow's ZORI (Zillow Observed Rent Index), which provides time series data for average rent prices in given zip codes. I filtered this dataset to focus on two key metros:

1. San Francisco-Oakland-Berkeley
2. San Jose-Sunnyvale-Santa Clara

This left me with time series rent averages for Bay Area zip codes.

Yelp Data  
With the zip codes from the ZORI dataset, I queried the Yelp business search API. For each zip code, I queried for certain types of businesses such as:
- Matcha
- Bubble tea / boba
- Coffee
- Pilates / yoga
- Etc.

For each business, we get the following types of info:
- Business name
- Rating
- Review count
- Price
- Etc.

### Feature Engineering
Ground Truth  
I created ground truth gentrification class labels by analyzing the ZORI dataset. The top 25% of rent growth was classified as "gentrified." This resulted in zip codes that had a 19%–66% increase in average rent prices.

Gentrification Scoring  
1. Price Tier  
    1. Price tiers were used for gentrification score. Each `$` in the Yelp price rating was another point towards the gentrification score. 
2. Ratings  
    1. An average rating over 4 stars received one point  
    2. An average rating over 4.5 stars received two points  
3. Engagement  
    1. High engagement with a business positively weighted the gentrification score  
4. Key Words  
    1. Key words such as `artisanal`, `craft`, etc. were used

### Model Implementation and Performance
I implemented the following three models:
1. Logistic Regression
2. Gradient Boosting
3. Random Forest

Results
```
              model   accuracy   precision   recall     f1         roc_auc
Logistic Regression   0.681818   0.428571    0.500000   0.461538   0.645833
      Random Forest   0.727273   0.500000    0.500000   0.500000   0.666667
  Gradient Boosting   0.727273   0.500000    0.333333   0.400000   0.677083
```
Out of the three models, Gradient Boosting performed the best with the highest accuracy and ROC.

![ROC](./outputs/roc_curves.png)

### Feature Importance
Predictors such as business quality, average rating, premium business density, as well as the previously discussed gentrification signals, were important features for the model.

![feature importance](./outputs/classification_feature_importance.png)

### Maps
> Please open the gentrification map HTML file in the output dir in the browser for an interactive map

Preview  
![preview](./outputs/gentrification-map-preview.png)

This map shows us in red the high-growth, truly gentrified neighborhoods. For each point here, we are also provided a gentrification density index.

An interesting note is that this map highlights areas such as Concord, Brentwood, and certain zip codes in San Francisco. My initial assumptions were that SF, Berkeley, and other metropolitan areas would have higher rates, but upon further research, I learned that these areas have had these types of businesses and rent has been high but stable. These other regions are developing and thus their rent change will be higher. I tested this against the Warm Springs region in Fremont. This zip code showed up as red with a high gentrification index. This stands true to what I know about the region—lots of new development and lots of new gentrified businesses.

### Discussion
Data Model Issues  
My model dealt with overfitting issues since I did not have enough data despite having a very large initial dataset. Since I focused on Bay Area zip codes, I was limited to ~100 valid zip codes. When I created my train/test split, these splits did not have enough data and the model overfit.

One solution to prevent this issue is to expand to more zip codes. Areas such as New York and Los Angeles would have expanded the zip code regions and the ZORI data was available. The issue with this arises from gathering the Yelp data. Since I was querying for specific types of businesses, I relied on the API. Querying the businesses for the existing zip codes required a hefty number of API calls, which is time consuming, and I exhausted my monthly base tier API calls. Expanding this would be expensive and time consuming.

### Future Improvements
To improve this model, I would need to improve my data model.
1. More zip codes across more metros would result in a more generalizable model. 
2. Yelp time series data that shows the change in neighborhoods over time and the change over time in these performative businesses.

### Applications
A model like this that can analyze neighborhood businesses to predict gentrification can be a powerful tool in predicting rent price changes over time. This model would enable us to monitor businesses as a gentrification indicator and could serve as an early warning sign for displacement risk. Additionally, people in real estate could use a model like this to find up-and-coming zip codes to invest in.