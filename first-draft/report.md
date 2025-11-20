With this project, I detracted from my tradtional approach of a gentrification classification - instead I pivoted towards creating a Bay Area community recommender. The general principle is as follows: can we combine Yelp adn Zillow data to find neighborhoods in the Bay Area with trending / performative businesses that will be entertaining for people in their 20s while minimizing rent prices.

I started with a heutisitic performed this minimization - it created an attraction score based on the businesses in the area and weighted this with rent prices. Zip codes that served this well weere recommender. 

Output
Zip codes in SF, Oakland, and Berkely as expected served the highest recommendations, but I beleive that my terms for the attractive businesses were too restrictign as certain communities had very low attraction scores as seen by the recommendations map. For future work, I should rehting this logic. 

Future work
This week, I was unable to see how I could effectively create a train / test split for this data. As I don't have any ground truth for thsi data, I found evaluation metrics difficult for thsi problem as well. I need to rethink how I plan on continuning to fit this problem into a more ML lens. 
