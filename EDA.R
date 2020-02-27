library(tidyverse)
library(readr)
library(ggplot2)
library(dplyr)
library(wordcloud)
library(RColorBrewer)

##hello 
###############Sarah################3

options(scipen=200)

## load datasets
alldata <- read.csv("alldata.csv")

all <- alldata %>% 
  select(position, company, location)
View(all)

wage = read_csv("City_of_Seattle_Wages___Comparison_by_Gender__Wage_Progression_Job_Titles.csv")
View(wage)

house <- read.csv("house_sales.csv")
View(house)

## Data Cleaning 
colnames(wage)[colnames(wage)=="Job Classification"] <- "Job_Classification"
colnames(wage)[colnames(wage)=="Total Avg Hrly Rate"] <- "Total_Avg_Hrly_Rate"
wage2 = wage[grep("Anlyst",wage$Job_Classification),]

## re-format the location column and add a new column called 'state'
city_trim <- function(s) gsub('([A-z]+, [A-Z]{2})(\ [0-9]{5})', '\\1', s)
all$location <- sapply(all$location, city_trim)
state_extract <- function(s) gsub('^([A-z| ]+, )([A-Z]{2}).*', '\\2', s)
all$state <- sapply(all$location, state_extract)
dim(all) #6964 x 4
View(all)

## Companies with Most Data Science Job Openings
com <- all %>%
  group_by(company)%>%
  count()%>%
  arrange(desc(n))

com %>%
  head(10)%>%
  ggplot(aes(x=reorder(company,n),y=n)) +
  geom_col(fill="#6391e0") + 
  coord_flip() +
  labs(x="", y="Number of Job Openings")+
  geom_text(aes(label=n), y=80, size=4, color="black")

## Top 5 City with the Largest Number of Data Scientist Jobs
city_counts <- all %>%
  group_by(location) %>%
  summarize(counts = n()) %>%
  arrange(-counts) %>% 
  top_n(5)

ggplot(city_counts, aes(x=reorder(location, -counts), y = counts, fill = location)) +                
  geom_bar(stat="identity") +
  geom_text(aes(label = paste0(counts))) +
  labs(x = "City", y = "Number of Data Scientist Jobs")

## Top 10 analyst hourly wages
wage2 %>% 
  select(Job_Classification,Total_Avg_Hrly_Rate) %>% 
  arrange(desc(Total_Avg_Hrly_Rate)) %>% 
  top_n(10)->top10_hourly_rate

ggplot(top10_hourly_rate,aes(x=reorder(Job_Classification,-Total_Avg_Hrly_Rate), y=Total_Avg_Hrly_Rate))+
  geom_col(aes(fill=Job_Classification))+
  labs(y="Avg Hourly Rate", x="Job Classification")+
  geom_text(aes(label=Total_Avg_Hrly_Rate))+
  theme(axis.text = element_text(size=10,face="italic",angle = 25))

## Working as a data scientist, would you be able to afford buying a house in Seattle?
ggplot(data = house, aes(x = house$sqft_living, y = house$price, color = "#6391e0")) +
  geom_point() +
  ggtitle('Relationship Between House Price and Sqft in Seattle') +
  labs(x = 'House Sqft', y = 'Price', caption = 'Data source: Kaggle Seattle House Sales Data')

## Price and grading
ggplot(data = house, aes(x = house$grade, y = house$price)) +
  geom_point(color = "#00AFBB", size = 2) +
  stat_smooth(color = "#FC4E07", fill = "#FC4E07",method = "loess") +
  ggtitle('Relationship Between House Price and Grading in Seattle') +
  labs(x = 'Grade', y = 'Price', caption = 'Data source: Kaggle Seattle House Sales Data')

