# =======================================================================================================================
#   =======================================================================================================================
#     
#     Job_Title:		Title of Role
#   Link:			Weblink of Job Posting
#   Queried_Salary:		Salary Range of the Job Posting (Estimated/Actual if available)
#   Job_Type:		3 Categories of Job Types - data_scientist, data_analyst, data_engineer
#   Skill:			List of desired skills on indeed site
#   No of Skill:		Count of the number of desired skills
#   Company:		Company that posted the job posting
#   No of Reviews:		Number of Reviews for the Company
#   No of Stars:		Ratings for the Company
#   Date Since Posted:	Number of days since the job was posted - if less than a day, will be rounded up to a full day
#   Description:		Web scrape of part of the job description
#   Location:		State the job opening is located in
#   Company_Revenue:	Annual revenue of hiring company
#   Company_Employees:	Employee count of hiring company
#   Company_Industry:	Industry of hiring company
#   
#   =======================================================================================================================
#     =======================================================================================================================
#     
#     Expansion of Skill column (Top 10 overall skills):
#     - python
#   - sql
#   - machine learning
#   - r
#   - hadoop
#   - tableau
#   - sas
#   - spark
#   - java
#   - Others
#   
#   Expansion of Location column (Top 10 overall states):
#     - CA
#   - NY
#   - VA
#   - TX
#   - MA
#   - IL
#   - WA
#   - MD
#   - DC
#   - NC
#   - Other_states
#   
#   Expansion of Company_Industry columns (Top 5 overall industries)
#   - Consulting and Business Services
#   - Internet and Software
#   - Banks and Financial Services
#   - Health Care
#   - Insurance
#   - Other_industries


# Goal: Create predictive models for salary prediction 

# Set working directory 
# setwd("~/Desktop")

# load libraries
library(dplyr)
library(ggplot2)
library(fastDummies)
library(caret)
library(MASS)
library(kernlab)
library(randomForest)
library(gbm)


# load the dataset 
data <- read.csv("indeed_job_dataset.csv")
glimpse(data)

########## Create a new working data called my data
# remove some columns 
mydata <- data %>% select(-X:-Link, -Skill, -Company, -Date_Since_Posted:-Location, -Company_Industry)
dim(mydata)

########## EDA
head(mydata)
summary(mydata) 
# 3 main job types: analyst, engineer, scientist 
# No. of skills: Median - 7, Mean - 7.804, Range - 0 - 20
# 962 companies don't have any reviews/ ratings on Indeed
# Ineed does not have information on some companies revenue and number of employee information 

#
levels(mydata$salary)
percentage <- prop.table(table(mydata$salary)) * 100
cbind(freq=table(mydata$salary), percentage=percentage)


# Count of each salary range
ggplot(mydata) + 
  geom_histogram(aes(x = as.factor(Queried_Salary)),stat="count") + 
  theme_classic() + 
  labs(title = "Distribution of Estimated / Actual Salary Range of the Job Postings",
       x = "Salary Range", y = "Frequency")

# % of each salary range among the dataset 
mydata %>% 
  group_by(Queried_Salary) %>% 
  summarize(count=n()) %>% 
  mutate(perct = round(prop.table(count),2)*100) %>% 
  ggplot(aes(x = Queried_Salary, y = perct)) +
  geom_histogram(stat = "identity")+
  geom_text(aes(x=Queried_Salary, y=0.01, label= sprintf("%.2f%%", perct)),
            hjust=0.5, vjust=-3, size=4,
            color="white", fontface = "bold") +
  theme_classic() +
  labs(x = "Salary Range", y="Percentage (%)",
       title = "Estimated / Actual Salary Range of the Job Postings (%)")

# Count of each job type 
mydata %>% 
  group_by(Job_Type) %>% 
  summarize(count = n()) %>% 
  ggplot(aes(x = Job_Type, y = count)) + 
  geom_bar(stat = "identity") + 
  theme_classic() + 
  geom_text(aes(x = Job_Type, y = 1, label = count),
            hjust = 0.5, vjust = -3, size = 4,
            color = "white", fontface = "bold") +
  labs(title = "Distribution of Job Types", x = "Job Type", y="Count") 

## Multivariate Plots - look at the interactions between the variables
skills <- mydata[ ,8:17] 
skills %>% head()
featurePlot(x=skills, y=mydata$Queried_Salary, plot="box")


########## Data Cleaning 
#mydata <- data %>% select(-X:-Link, -Skill, -Company, -Date_Since_Posted:-Location, -Company_Industry)

summary(mydata) # shows that Company_Revenue & Company_Employees have blank values 
# fill those blank value with NA 
# Company_Revenue
mydata$Company_Revenue <- as.character(mydata$Company_Revenue)
mydata$Company_Revenue[mydata$Company_Revenue == ""] <- "NA"
mydata$Company_Revenue <- as.factor(mydata$Company_Revenue)
summary(mydata$Company_Revenue)

mydata$Company_Employees <- as.character(mydata$Company_Employees)
mydata$Company_Employees[mydata$Company_Employees == ""] <- "NA"
mydata$Company_Employees <- as.factor(mydata$Company_Employees)
summary(mydata$Company_Employees)


# Check if there’s any missing value in this dataset
sapply(mydata, function(x) sum(is.na(x)))

# replace NAs with o for No_of_Reviews & No_of_Stars
mydata[is.na(mydata)] <- 0

# Dummify the following columns 
str(mydata) # check if the columns needed to be dumified are in factor forms 
mydata <-dummy_cols(mydata)

mydata <- mydata %>% select(-Job_Type, -Company_Revenue, - Company_Revenue, - Company_Employees,
                            -"Queried_Salary_<80000": -"Queried_Salary_80000-99999" )

# change colnames 
colnames(mydata)
mydata <- mydata %>% rename_all(tolower)

colnames(mydata)[colnames(mydata) == "queried_salary"] <- "salary"
colnames(mydata)[colnames(mydata) == "others"] <- "other_skills"

colnames(mydata)[colnames(mydata) == "ca"] <- "california"
colnames(mydata)[colnames(mydata) == "ny"] <- "new_york"
colnames(mydata)[colnames(mydata) == "va"] <- "virginia"
colnames(mydata)[colnames(mydata) == "tx"] <- "texas"
colnames(mydata)[colnames(mydata) == "ma"] <- "massachusetts"
colnames(mydata)[colnames(mydata) == "il"] <- "illinois"
colnames(mydata)[colnames(mydata) == "wa"] <- "washington"
colnames(mydata)[colnames(mydata) == "md"] <- "maryland"
colnames(mydata)[colnames(mydata) == "dc"] <- "dc"
colnames(mydata)[colnames(mydata) == "nc"] <- "north_carolina"

colnames(mydata)[colnames(mydata) == "job_type_data_analyst"] <- "data_analyst"
colnames(mydata)[colnames(mydata) == "job_type_data_engineer"] <- "data_engineer"
colnames(mydata)[colnames(mydata) == "job_type_data_scientist"] <- "data_scientist"


colnames(mydata)[colnames(mydata) == "company_revenue_$1b to $5b (usd)"] <- "revenue_$1bto$5b"
colnames(mydata)[colnames(mydata) == "company_revenue_$5b to $10b (usd)"] <- "revenue_$5bto$10b"
colnames(mydata)[colnames(mydata) == "company_revenue_less than $1b (usd)"] <- "revenue<$1b"
colnames(mydata)[colnames(mydata) == "company_revenue_more than $10b (usd)"] <- "revenue>$10b"
colnames(mydata)[colnames(mydata) == "company_revenue_na"] <- "revenue_na"

colnames(mydata)[colnames(mydata) == "company_employees_10,000+"] <- "employees>10k"
colnames(mydata)[colnames(mydata) == "company_employees_less than 10,000"] <- "employees<10k"
colnames(mydata)[colnames(mydata) == "company_employees_na"] <- "employees_na"
colnames(mydata)


################ Relabel the levels of salary to a value 
levels(mydata$salary)


################ Split into the training and testing datasets
