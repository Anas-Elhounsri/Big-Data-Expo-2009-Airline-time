# Big Data Sparks: Expo 2009 Airline Time
A Big Data project were we predict the possible time delay for flights using the [Data Exp 2009: Airline dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7).
## Variables used:
Following a comprehensive analysis of the dataset, we have strategically identified a set of numerical and categorical variables to serve as the foundation for our Machine Learning algorithms.

### Numerical variables:
* __Month:__ This variable holds a significant importance in predicting arrival delays as it encapsulates seasonal patterns inherent in air travel. Incorporating this variable into our model allows it to discern and comprehend the cyclicality of delays influenced by different times of the year.
* __DayofMonth, DayOfWeek:__ Similar to “Month”, these two variables tend to have a seasonal trend that would reflect holiday and weekend patterns, which the model can potentially learn the patterns related to specific days of the month and week.
* __DepTime, CRSDepTime:__ These variables describe the departure time and the scheduled departure time, and it can have insights on time sensitive factors such as morning rush hours, late night flight, which can contribute to “ArrDelay”.
* __CRSArrTime:__ This variable described scheduled arrival time, which is very important, as the machine learning can use this variable as a leverage along with other variables like DepTime & CRSDepTime to capture the overall flight schedule dynamics. Delays at earlier stages can ripple through the schedule impacting “ArrDelay”.
* __TaxiOut:__ This variable represents the time the plane was able to take off, delays during this procedure can signify operational inefficiencies, and including the variable galps the model account for these factors in predicting arrival delays.
