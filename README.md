# Omid Ghamiloo
## Master of Data Science at Sapienza university
# IMDB Scarping
### This dataset is collected from IMDB and uses the official website directly and includes Feature Film/TV Movie that has been Released between 1900-01-01 and 2020-06-01 by BeautifulSoup library on python3. 
## Data collecting in this dataset has 3 steps. 
### -First Step: After you search for the popular movies which are released between 1900 and 2020, there are 4924 pages on IMDB and each page introduces 100 movies. The first step is collecting these 4924 pages beacuse each page has the unique and different web address. There are many ways for collecting data like this Definitely, but because this is my first project on Web Scraping, I did this.
### -Second Step: As I metioned before, each page that is collected in the previous step, has 100 movies. In this step the link of each movie is collected. There are many information about the movies on the  main page with 100 movies, but I need some detail, So I should to collect the data from the main page of the movies. At the end of this step, there are 492100 links of popular movies.
### -Third Step: The Web scraping subject is totally in this step that collect the information about each movie. This infromation includes MovieID, Title, Original Title, Released date,Country, Location, Languages, Directors, Writers, Stars, Gross, Run Time, Rate, Number of votes, Genres and summary.
### The first and second step have been done on my laptop with core i5 CPU and 16 GB of Ram in jupyter notebook with python 3
### The third part has been done in SageMaker service with ml.t2.medium instance that is developed by Amazon AWS. 
### Web Scarping : Web scraping, web harvesting, or web data extraction is data scraping used for extracting data from websites.
