# Course project: predict type of movement on Weight Lifting Exercises dataset
TizVic  
May 17, 2017  



# Executive summary
In this document I analyze the **Weight Lifting Exercises dataset** (see <http://groupware.les.inf.puc-rio.br/har>) to investigate how an activity was performed by six young man. They were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: class A corresponds to the specified execution of the exercise, while the other 4 classes (B - E) correspond to common mistakes. 

To determine which class (A-E) correspond to the sensor signals I use a Random Forest classification algorithm after selecting it between different algorithms by means of Accuracy analysis. The algorithm has an in-sample error of 0.17% and an expected out-of-sample error of 0.28%  in the classification of the movements. Finally, I use the algorithm to predict the result on a test set unlabeled and results are compatible with expected out-of-sample error.

# Data processing
## Download datasets
The datasets were downloaded from the [address](https://www.coursera.org/learn/practical-machine-learning/peer/R43St/prediction-assignment-writeup) indicated in the assignment and loaded in memory using the following code:

```r
## Load libraries
require(caret)
require(e1071)
require(gbm)
require(randomForest)
require(kernlab)
```

```r
## Download TRAINING SET
if (!file.exists("./data/pml-training.csv")) {
        if (!dir.exists("./data")) {
                dir.create("./data")
        }
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                      "./data/pml-training.csv")
}
## Download TESTING SET
if (!file.exists("./data/pml-testing.csv")) {
        if (!dir.exists("./data")) {
                dir.create("./data")
        }
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                      "./data/pml-testing.csv")
}

## Load data
training <- read.csv("./data/pml-training.csv",
                     stringsAsFactors = F, 
                     header = T,
                     na.strings = c("NA","","#DIV/0"))
qzTesting <- read.csv("./data/pml-testing.csv",
                    stringsAsFactors = F,
                    header = T)
```

## Data preparation
First I analyzed how the data is structured and whether there are NAs

```r
str(training[,1:15])
```

```
## 'data.frame':	19622 obs. of  15 variables:
##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name           : chr  "carlitos" "carlitos" "carlitos" "carlitos" ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp      : chr  "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" ...
##  $ new_window          : chr  "no" "no" "no" "no" ...
##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt  : chr  NA NA NA NA ...
##  $ kurtosis_picth_belt : chr  NA NA NA NA ...
##  $ kurtosis_yaw_belt   : chr  NA NA NA NA ...
##  $ skewness_roll_belt  : chr  NA NA NA NA ...
```

```r
# how many NAs per column?
list <- apply(training, 2, function (x) {sum(is.na(x))})/nrow(training)
sprintf("Number of column with less then 50 perc of significative data: %i", sum(list > .5))
```

```
## [1] "Number of column with less then 50 perc of significative data: 100"
```
There are many columns with more NAs than data so I remove these column; in addition the first six column relate to data not interesting for analysis like name of subject, date and time of event so I remove these also:  

```r
# remove column with too many NAs (more than 50% NAs) and first 6 column useless
col2Remove <- which(list > .5, useNames = F)
train2 <- training[,-c(1:6,col2Remove)]
qzTest <- qzTesting[,-c(1:6,col2Remove)]
```

Last check for columns with near zero variance predictors:

```r
# remove near zero column
col2Remove <- nearZeroVar(train2)
sprintf("Column with near zero variance predictors: %i", length(col2Remove))
```

```
## [1] "Column with near zero variance predictors: 0"
```
and final data preparation

```r
# Final preparation
train1 <- train2
train1$classe <- as.factor(train1$classe)
str(train1)
```

```
## 'data.frame':	19622 obs. of  54 variables:
##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
##  $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

## Data splitting
Now that the dataset is ready to use I divide it in three subsets: training set, cross validation set and test set. I chose this subdivision because the source dataset is quite large and this allows me to have a set for training (`ltTR`), where the various algorithms are trained, one set to check the performances (`ltCV`), unrelated to the first one, and finally a set (`ltTS`), completely independent of the two previous ones, on which to test the expected out-of-sample errors.

```r
## for reproducibility
set.seed(234)

# creation TRAIN/CV/TEST sets (60/20/20)
inTrain <- createDataPartition(train1$classe, p=.6, list = F)
inCV <- createDataPartition(train1[-inTrain,]$classe, p=.5, list = F)
ltTR <- train1[inTrain,]
ltCV <- train1[-inTrain,][inCV,]
ltTS <- train1[-inTrain,][-inCV,]
sprintf("Size TRAIN SET: %i, CV SET: %i, TEST SET: %i",
        dim(ltTR)[1],
        dim(ltCV)[1],
        dim(ltTS)[1])
```

```
## [1] "Size TRAIN SET: 11776, CV SET: 3923, TEST SET: 3923"
```

# Model selection
## Introduction
Let's start by analyzing the data using different classification algorithms starting from the simpler ones and proceeding to the more complex ones.
All these algorithms are trained on the train set (`ltTR`) while accuracy is calculated on the cv set (`ltCV`). For each of the algorithms, the processing time is calculated so that it can also be estimated the cost in cpu time.

## Linear discriminant analysis
See <https://en.wikipedia.org/wiki/Linear_discriminant_analysis> for reference:

```r
# LDA
ltTRlda <- ltTR
ltCVlda <- ltCV
# Train on ltTR set
ldaTime <- system.time(modFitLDA <- train(classe ~ ., 
                                          method = "lda", 
                                          preProc = c("center", "scale"),
                                          data = ltTRlda))
# prediction and accuracy on ltCV set
resLDA <- predict(modFitLDA, ltCVlda)
accLDA <- confusionMatrix(resLDA, ltCVlda$classe)$overall['Accuracy']
dfTime <- data.frame(algorithm="LDA",
                     accuracy=accLDA,
                     processTime=ldaTime[3], stringsAsFactors = F)
sprintf("LDA accuracy: %.3f, process time (total s): %.2f",
        accLDA,
        ldaTime[3])
```

```
## [1] "LDA accuracy: 0.711, process time (total s): 13.62"
```
## Support Vector Machines algorithm (SVMs)
For the SVMs algorithm I used two type of kernel: linear and radial (see <https://en.wikipedia.org/wiki/Support_vector_machine> for reference):
### Linear kernel SVM

```r
## SVM Linear Kernel
ltTRsvm <- ltTR
ltCVsvm <- ltCV
linTime <- system.time(
                        modFitSVML <- train(classe ~ ., 
                                           method = "svmLinear", 
                                           preProc = c("center","scale"),
                                           data = ltTRsvm)
                )
resSVML <- predict(modFitSVML, ltCVsvm)
accSVML <- confusionMatrix(resSVML, 
                          ltCVsvm$classe)$overall['Accuracy']
dfTime <- rbind(dfTime, list("Linear kernel SVM", accSVML, linTime[3]))
sprintf("SVM Linear kernel accuracy: %f,  process time (total s): %f",
        accSVML,
        linTime[3])
```

```
## [1] "SVM Linear kernel accuracy: 0.789957,  process time (total s): 288.190000"
```

### Radial kernel SVM

```r
## SVM Radial Kernel
radTime <- system.time(modFitSVMR <- train(classe ~ ., 
                                           method = "svmRadial", 
                                           preProc = c("center","scale"),
                                           data = ltTRsvm))
resSVMR <- predict(modFitSVMR, ltCVsvm)
accSVMR <- confusionMatrix(resSVMR, 
                          ltCVsvm$classe)$overall['Accuracy']
dfTime <- rbind(dfTime, list("Radial kernel SVM", accSVMR, radTime[3]))
sprintf("SVM Radial kernel accuracy: %f,  process time (total s): %f",
        accSVMR,
        radTime[3])
```

```
## [1] "SVM Radial kernel accuracy: 0.925057,  process time (total s): 2343.780000"
```

### SVM considerations
Radial kernel it is better but at much higher cost of CPU time.

## Generalized Boosted Regression Modeling
For boosting see <https://en.wikipedia.org/wiki/Boosting_(machine_learning)>.

```r
## Boosting with trees
ltTRgbm <- ltTR
ltCVgbm <- ltCV
gbmTime <- system.time(
        modFitGBM <- train(classe ~ ., 
                           method = "gbm", 
                           preProc = c("center","scale"),
                           train.fraction = .99,
                           verbose = F,
                           data = ltTRgbm)
)
resGBM <- predict(modFitGBM, ltCVgbm)
accGBM <- confusionMatrix(resGBM, 
                           ltCVgbm$classe)$overall['Accuracy']
dfTime <- rbind(dfTime, list("GBM", accGBM, gbmTime[3]))
sprintf("GBM accuracy: %f, process time (total s): %f",
        accGBM,
        gbmTime[3])
```

```
## [1] "GBM accuracy: 0.978843, process time (total s): 1151.480000"
```

## Random forest
For reference see <https://en.wikipedia.org/wiki/Random_forest>.

```r
## Random Forest
ltTRrf <- ltTR
ltCVrf <- ltCV
rfTime <- system.time(modFitRF <- randomForest(classe ~ ., 
                                               method = "rf", 
                                               data = ltTRrf))
resRF <- predict(modFitRF, ltCVrf)
accRF <- confusionMatrix(resRF, 
                         ltCVrf$classe)$overall['Accuracy']
dfTime <- rbind(dfTime, list("RandomForest", accRF, rfTime[3]))
sprintf("RF accuracy: %f, process time (total s): %f",
        accRF,
        rfTime[3])
```

```
## [1] "RF accuracy: 0.998216, process time (total s): 29.630000"
```
## Final selection
Summarizing the results seen in the previous subsections in the following table:

algorithm         |    accuracy |  processTime
------------------|  ---------- | ------------
LDA               |   0.7106806 |        13.94
Linear kernel SVM |   0.7899567 |       288.19
Radial kernel SVM |   0.9250574 |      2343.78
GBM               |   0.9788427 |      1151.48
RandomForest      |   0.9982157 |        29.63

The two best algorithms on cross validation test are GBM and Random Forest with very similar performance but very different processing times so I choose Random Forest as the final model.


# Error evaluation
To estimate out-of-sample errors of the final model I can not use the cross validation set because I used it for choosing the algorithm and therefore there are risk overfitting.
The evaluation of accuracy on the test set `ltTS`, that is completely independent of the previous ones, is:

```r
predErr <- predict(modFitRF, ltTS)
confusionMatrix(predErr, ltTS$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  756    2    0    0
##          C    0    2  682    3    0
##          D    0    0    0  640    3
##          E    0    0    0    0  718
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9972         
##                  95% CI : (0.995, 0.9986)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9965         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9960   0.9971   0.9953   0.9958
## Specificity            0.9996   0.9994   0.9985   0.9991   1.0000
## Pos Pred Value         0.9991   0.9974   0.9927   0.9953   1.0000
## Neg Pred Value         1.0000   0.9991   0.9994   0.9991   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1927   0.1738   0.1631   0.1830
## Detection Prevalence   0.2847   0.1932   0.1751   0.1639   0.1830
## Balanced Accuracy      0.9998   0.9977   0.9978   0.9972   0.9979
```
The accuracy of the model 0.997196 on the test set is lower then the accuracy on cv set 0.9982157 as expected, but is however very high and this could be a synthom of overfitting. In a dataset with *different subjects*, which will have different ways to perform the various types of exercises, we can expect a lower accuracy.

I applied the model to the testing dataset provided and the responses were:

```r
# predict for final quiz 
resQZ <- predict(modFitRF, qzTest)
print(resQZ)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
These responses inputted in the final quiz have given a score of 20/20, compatible with expected accuracy, taking into account the small number of data in the dataset where each answer counts for 1/20 of the final result.


