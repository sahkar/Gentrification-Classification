# Gentrification Classifier
By: Sahith Karra

### Research Question
Recently, there has been a trend with "performative" males that engage in activies such as drinking matcha lattes, reading feminist literature, using wired earbuds, and more. When I travelled to San Francisco and other parts of the Bay, I realized how many people follow this archtype either ironically or unironically. This thinking lead me to another school of thought - how different Bay Area landscapes area. There are old dilapidated neighborhoods, suburban communities, and trendy areas where I could find these "performative" males. 

This made me ask the core question of this research experiment - could a given neighborhood's business type be a predictor of rent price increase. Bascially, could the influx of trendy and hip businesses such as matcha cafes, yoga studios, KBBQ resturants, and etc be a sign that that the neighborhood is gentrified. 

> Gentrification: The process of transforming a lower-value urban neighbordhood into a higher-value one through the influx of wealthier resident typically characterized by the increase in cost of living. 

### Data Sources
Rent Prices
Rather than find a shallow "gentrification" dataset with ambigious class labels, I crafted my own gentriciation labels using historic rent price data. I relied on Zillow's ZORI (Zillow Observed Rent Index) which provides time series data for average rent prices in a given zip codes. I filtered this dataset to focus on two key metros

1. San Francisco-Oakland-Berkeley
2. San Jose-Sunnyvale-Santa Clara

This left me with time series rent averages for Bay Area zip codes. 

Yelp Data
With the zip codes from the ZORI dataset, I query the Yelp business search API. For each zip code, I query for certain types of businesses suach as 
- Matcha
- Bubbletea / boba
- Coffee
- Pilates / yoga
- Etc

For each business, we get the following types of info 
- Business name
- Rating
- Review count
- Price
- Etc

### Feature Engineering
Ground Truth 
I created ground truth gentrification class labels by analyzing the ZORI dataset. The top 25% of rent growth was classified as "gentrified". This resulted in zip codes that had a 19% - 66% increase in average rent prices. 

GentrificatioN Scoring
1. Price Tier 
    1. Price tiers were used for gentrification score. Each `$` in the Yelp price rating was another point towards gentrifications core. 
2. Ratings
    1. An average rating over 4 stars received one point
    2. An average rating over 4.5 stars received two points
3. Engagement
    1. High engagement with a businesses positively weighted gentrification score
4. Key Words
    1. Key words such as `artisinal`, `craft`, etc were used






