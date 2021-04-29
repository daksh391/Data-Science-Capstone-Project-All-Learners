
#####################################################################################################################################
#Pre-processing

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("DT", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(DT)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Extract year of release and year of rating
edx <- edx%>%mutate(ReleaseYear =  as.numeric(str_sub(title,-5,-2)))
edx <- edx%>% mutate(RatingYear = year(as_datetime(timestamp)))
edx<-edx%>%select(-c(timestamp, title))

validation <- validation%>%mutate(ReleaseYear =  as.numeric(str_sub(title,-5,-2)))
validation <- validation%>% mutate(RatingYear = year(as_datetime(timestamp)))
validation<-validation%>%select(-c(timestamp, title))


#####################################################################################################################################
#Exploratory Data Analysis
#Summary
edx %>%
  summarize(Nr_Users = n_distinct(userId),
            Nr_Movies = n_distinct(movieId),
            Nr_Ratings = n(),
            Nr_Genres = n_distinct(genres),
            Nr_ReleaseYears = n_distinct(ReleaseYear),
            Nr_RatingYears = n_distinct(RatingYear))

#Movies Distribution
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, color = "black", fill="darkblue") + 
  scale_x_log10() + 
  labs(title="Movies Distribution",x="Nr Ratings by Movie (log10)", y = "count")

#Users Distribution
edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, color = "black", fill = "darkblue") + 
  scale_x_log10() + 
  labs(title="Users Distribution",x="Nr Ratings by User (log10)", y = "count")

#Genre Distribution
edx %>% 
  dplyr::count(genres) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, color = "black", fill = "darkblue") + 
  scale_x_log10() + 
  labs(title="Genres Distribution",x="Nr Ratings by Genres (log10)", y = "count")

#Release Year Distribtion
edx%>%ggplot(aes(ReleaseYear))+
  geom_histogram(bins = 50, color = "black", fill = "darkblue") + 
  labs(title="Release Year Distribution",x="Release Year", y = "count")

#Rating Year Distribtion
edx%>%ggplot(aes(RatingYear))+
  geom_histogram(bins = 50, color = "black", fill = "darkblue") + 
  labs(title="Rating Year Distribution",x="Rating Year", y = "count")

#Ratings Distribution
edx%>%ggplot(aes(rating))+
  geom_histogram(col='black',bins=10, fill = "darkblue")+
  scale_x_continuous(breaks = seq(0.5,5,0.5))+
  geom_vline(xintercept =mean(edx$rating), color="red", linetype="dashed", size=1)+ 
  labs(title="Ratings Distribution",x="Rating", y = "count")

#####################################################################################################################################
#Model Training
# Splitting edx dataset into training and test datasets
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


#Model 1: Average
mu <- mean(train_set$rating)
Model1_RSME <- RMSE(test_set$rating, mu)
Models_RMSE <- tibble(Model = "Model 1: Average", RMSE = Model1_RSME)
datatable(Models_RMSE)

#Model 2: Movie Effect
Movie_Bias <- train_set %>%  #Calculating movie bias
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - mu))

Model2_predicted_ratings <- mu + test_set %>%  #Testing the model
  left_join(Movie_Bias, by='movieId') %>%
  .$b_m

Model2_RMSE <- RMSE(test_set$rating, Model2_predicted_ratings) # Evaluating the RMSE of the model

Models_RMSE  <- rbind(Models_RMSE, c("Model 2: Movie Effect", Model2_RMSE)) #Storing the RMSE
datatable(Models_RMSE)

#Model 3: Movie + User Effect
User_Bias <- train_set %>%  #Calculating user bias
  left_join(Movie_Bias, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))

Model3_predicted_ratings <- test_set %>%  #Testing the model
  left_join(Movie_Bias, by='movieId') %>%
  left_join(User_Bias, by='userId') %>%
  mutate(predicted_ratings = mu + b_m + b_u) %>%
  .$predicted_ratings

Model3_RMSE <- RMSE(test_set$rating, Model3_predicted_ratings) # Evaluating the RMSE of the model

Models_RMSE  <- rbind(Models_RMSE, c("Model 3: Movie + User Effect", Model3_RMSE)) #Storing the RMSE
datatable(Models_RMSE)

#Model 4: Movie + User + Release Year Effect
ReleaseYear_Bias <- train_set %>%  #Calculating release year bias
  left_join(Movie_Bias, by='movieId') %>%
  left_join(User_Bias, by='userId') %>%
  group_by(ReleaseYear) %>%
  summarize(b_y = mean(rating - mu - b_m - b_u))

Model4_predicted_ratings <- test_set %>% #Testing the model
  left_join(Movie_Bias, by='movieId') %>%
  left_join(User_Bias, by='userId') %>%
  left_join(ReleaseYear_Bias, by='ReleaseYear') %>%
  mutate(predicted_ratings = mu + b_m + b_u +b_y) %>%
  .$predicted_ratings

Model4_RMSE <- RMSE(test_set$rating, Model4_predicted_ratings) # Evaluating the RMSE of the model

Models_RMSE  <- rbind(Models_RMSE, c("Model 4: Movie + User + Release Year Effect", Model4_RMSE)) #Storing the RMSE
datatable(Models_RMSE)

#Model 5: Movie + User + Release Year + Genre Effect
Genre_Bias <- train_set %>%  #Calculating genre bias
  left_join(Movie_Bias, by='movieId') %>%
  left_join(User_Bias, by='userId') %>%
  left_join(ReleaseYear_Bias, by='ReleaseYear') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_m - b_u - b_y))

Model5_predicted_ratings <- test_set %>% #Testing the model
  left_join(Movie_Bias, by='movieId') %>%
  left_join(User_Bias, by='userId') %>%
  left_join(ReleaseYear_Bias, by='ReleaseYear') %>%
  left_join(Genre_Bias, by='genres') %>%
  mutate(predicted_ratings = mu + b_m + b_u +b_y +b_g) %>%
  .$predicted_ratings

Model5_RMSE <- RMSE(test_set$rating, Model5_predicted_ratings) # Evaluating the RMSE of the model

Models_RMSE  <- rbind(Models_RMSE, c("Model 5: Movie + User + Release Year +Genre  Effect", Model5_RMSE)) #Storing the RMSE
datatable(Models_RMSE)

#Model 6: Movie + User + Release Year + Genre + Rating Year Effect
RatingYear_Bias <- train_set %>%  #Calculating rating year bias
  left_join(Movie_Bias, by='movieId') %>%
  left_join(User_Bias, by='userId') %>%
  left_join(ReleaseYear_Bias, by='ReleaseYear') %>%
  left_join(Genre_Bias, by='genres') %>%
  group_by(RatingYear) %>%
  summarize(b_r = mean(rating - mu - b_m - b_u - b_y - b_g))

Model6_predicted_ratings <- test_set %>% #Testing the model
  left_join(Movie_Bias, by='movieId') %>%
  left_join(User_Bias, by='userId') %>%
  left_join(ReleaseYear_Bias, by='ReleaseYear') %>%
  left_join(Genre_Bias, by='genres') %>%
  left_join(RatingYear_Bias, by='RatingYear') %>%
  mutate(predicted_ratings = mu + b_m + b_u +b_y +b_g + b_r) %>%
  .$predicted_ratings

Model6_RMSE <- RMSE(test_set$rating, Model6_predicted_ratings) # Evaluating the RMSE of the model

Models_RMSE  <- rbind(Models_RMSE, c("Model 6: Movie + User + Release Year + Genre + Rating Year Effect", Model6_RMSE)) #Storing the RMSE
datatable(Models_RMSE)

#Model 7: Regularization using k-fold Cross validation
cv <- createFolds(train_set$rating, k=5,  list = TRUE, returnTrain = TRUE) #Splitting the training dataset into 5 for k-fold cross validation
lambdas <- seq(0, 10, 0.25)
ks <- seq(1, 5, 1)

rmses <- sapply(lambdas, function(l){ #function for applying various lamda values
  
  rmses_kfold<- sapply(ks, function(k){ #function for applying various k values
    mu <- mean(train_set$rating[-cv[[k]]])
    Movie_Bias_Reg <- train_set[-cv[[k]],] %>% #Calculating movie bias with regularization
      group_by(movieId) %>% 
      summarize(b_m_s = sum(rating - mu)/(n()+l))
    
    User_Bias_Reg <- train_set[-cv[[k]],] %>%  #Calculating user bias with regularization
      left_join(Movie_Bias_Reg, by='movieId') %>%
      group_by(userId) %>%
      summarize(b_u_s = sum(rating - mu - b_m_s)/(n()+l))
    
    ReleaseYear_Bias_Reg <- train_set[-cv[[k]],]%>% #Calculating release year bias with regularization
      left_join(Movie_Bias_Reg, by='movieId') %>%
      left_join(User_Bias_Reg, by='userId') %>%
      group_by(ReleaseYear) %>%
      summarize(b_y_s = sum(rating - mu - b_m_s - b_u_s)/(n()+l))
    
    Genre_Bias_Reg <- train_set[-cv[[k]],] %>%  #Calculating genre bias with regularization
      left_join(Movie_Bias_Reg, by='movieId') %>%
      left_join(User_Bias_Reg, by='userId') %>%
      left_join(ReleaseYear_Bias_Reg, by='ReleaseYear') %>%
      group_by(genres) %>%
      summarize(b_g_s = sum(rating - mu - b_m_s - b_u_s - b_y_s)/(n()+l))
    
    RatingYear_Bias_Reg <- train_set[-cv[[k]],] %>% #Calculating rating year bias with regularization
      left_join(Movie_Bias_Reg, by='movieId') %>%
      left_join(User_Bias_Reg, by='userId') %>%
      left_join(ReleaseYear_Bias_Reg, by='ReleaseYear') %>%
      left_join(Genre_Bias_Reg, by='genres') %>%
      group_by(RatingYear) %>%
      summarize(b_r_s = sum(rating - mu - b_m_s - b_u_s - b_y_s - b_g_s)/(n()+l))
  
    predicted_ratings <- train_set[cv[[k]],] %>% #Testing the model
      left_join(Movie_Bias_Reg, by='movieId') %>%
      left_join(User_Bias_Reg, by='userId') %>%
      left_join(ReleaseYear_Bias_Reg, by='ReleaseYear') %>%
      left_join(Genre_Bias_Reg, by='genres') %>%
      left_join(RatingYear_Bias_Reg, by='RatingYear') %>%
      mutate(predicted_ratings=  mu + b_m_s + b_u_s +b_y_s +b_g_s + b_r_s, predicted_ratings = coalesce(predicted_ratings, mu)) %>%
      .$predicted_ratings
    return(RMSE(predicted_ratings, train_set$rating[cv[[k]]])) #Evaluating the RMSE of the model
    })
  
  return(mean( rmses_kfold)) #Average of RMSE values
})

qplot(lambdas, rmses)  
l <- lambdas[which.min(rmses)] #Choosing lamda value with minimum rmse

mu <- mean(train_set$rating) #Calculating movie bias with regularization with chosen lamda value
Movie_Bias_Reg <- train_set%>% 
  group_by(movieId) %>% 
  summarize(b_m_s = sum(rating - mu)/(n()+l))

User_Bias_Reg <- train_set%>% #Calculating user bias with regularization with chosen lamda value
  left_join(Movie_Bias_Reg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u_s = sum(rating - mu - b_m_s)/(n()+l))

ReleaseYear_Bias_Reg <- train_set%>% #Calculating release bias with regularization with chosen lamda value
  left_join(Movie_Bias_Reg, by='movieId') %>%
  left_join(User_Bias_Reg, by='userId') %>%
  group_by(ReleaseYear) %>%
  summarize(b_y_s = sum(rating - mu - b_m_s - b_u_s)/(n()+l))

Genre_Bias_Reg <- train_set%>% #Calculating genre bias with regularization with chosen lamda value
  left_join(Movie_Bias_Reg, by='movieId') %>%
  left_join(User_Bias_Reg, by='userId') %>%
  left_join(ReleaseYear_Bias_Reg, by='ReleaseYear') %>%
  group_by(genres) %>%
  summarize(b_g_s = sum(rating - mu - b_m_s - b_u_s - b_y_s)/(n()+l))

RatingYear_Bias_Reg <- train_set%>% #Calculating rating year bias with regularization with chosen lamda value
  left_join(Movie_Bias_Reg, by='movieId') %>%
  left_join(User_Bias_Reg, by='userId') %>%
  left_join(ReleaseYear_Bias_Reg, by='ReleaseYear') %>%
  left_join(Genre_Bias_Reg, by='genres') %>%
  group_by(RatingYear) %>%
  summarize(b_r_s = sum(rating - mu - b_m_s - b_u_s - b_y_s - b_g_s)/(n()+l))

Model7_predicted_ratings <- test_set%>% #Testing the model
  left_join(Movie_Bias_Reg, by='movieId') %>%
  left_join(User_Bias_Reg, by='userId') %>%
  left_join(ReleaseYear_Bias_Reg, by='ReleaseYear') %>%
  left_join(Genre_Bias_Reg, by='genres') %>%
  left_join(RatingYear_Bias_Reg, by='RatingYear') %>%
  mutate(predicted_ratings=  mu + b_m_s + b_u_s +b_y_s +b_g_s + b_r_s, predicted_ratings = coalesce(predicted_ratings, mu)) %>%
  .$predicted_ratings


Model7_RMSE <- RMSE(test_set$rating, Model7_predicted_ratings) # Evaluating the RMSE of the model

Models_RMSE  <- rbind(Models_RMSE, c("Model 7: Regularized Movie + User + Release Year + Genre + Rating Year Effect (using k-fold cross-validation)", Model7_RMSE)) #Storing the RMSE
datatable(Models_RMSE)

#####################################################################################################################################
#Final model validation
Model7_predicted_ratings <- validation%>% 
  left_join(Movie_Bias_Reg, by='movieId') %>%
  left_join(User_Bias_Reg, by='userId') %>%
  left_join(ReleaseYear_Bias_Reg, by='ReleaseYear') %>%
  left_join(Genre_Bias_Reg, by='genres') %>%
  left_join(RatingYear_Bias_Reg, by='RatingYear') %>%
  mutate(predicted_ratings=  mu + b_m_s + b_u_s +b_y_s +b_g_s + b_r_s, predicted_ratings = coalesce(predicted_ratings, mu)) %>%
  .$predicted_ratings


Final_Model_RMSE<- RMSE(validation$rating, Model7_predicted_ratings)
Final_Model_RMSE