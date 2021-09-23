library(caret)
library(purrr)
library(class)

DB <- read.table(file="Merge_database.txt", header = TRUE)

# To ensure that there are NO  “zero- variance predictors”
ID <- rownames(DB)
a <- nearZeroVar(DB, freqCut = 95/5, uniqueCut = 10, saveMetrics = FALSE,
                 names = FALSE, foreach = FALSE, allowParallel = TRUE)
DB <- DB[,-a]
DB<- data.frame(sapply(DB, function(x) as.numeric(as.character(x))), 
                check.names=F)

# Remove variables with NAs
DB <- DB[,colSums(is.na(DB))==0]
rownames(DB) <- ID

## Remove attributes with high correlation
correlationMatrix <- cor(DB[,2:ncol(DB)])
highCorr <- sum(abs(correlationMatrix[upper.tri(correlationMatrix)]) > .999)

# summarize the correlation matrix
summary(correlationMatrix[upper.tri(correlationMatrix)])

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)

# remove those attributes
DB2 <- DB[,-(highlyCorrelated+1)]

# Downsampling of the groups in order to be balanced
DB2$Grupo <- as.factor(DB2$Grupo)
BD2 <- downSample(DB2, DB2$Grupo)

## create the Train and Test set
inTrain <- createDataPartition(y = BD2[,1], p = .6, list = FALSE)
training <- BD2[ inTrain,]
testing <- BD2[-inTrain,]


# To ensure that there are NO  “zero- variance predictors” II.
# I do it twice because the split of the samples can create new zero-variance attributes 
# and they won't be informative for the model and will introduce noise/confusion. In fact, the first
# removal is not necessary but we did it in order to simplify the removal of correlated attributes

training$Class <- NULL
testing$Class <- NULL
a <- nearZeroVar(training[,2:ncol(training)], freqCut = 95/5, uniqueCut = 10, saveMetrics = FALSE,
                 names = FALSE, foreach = FALSE, allowParallel = TRUE)
b <- nearZeroVar(testing[,2:ncol(training)], freqCut = 95/5, uniqueCut = 10, saveMetrics = FALSE,
                 names = FALSE, foreach = FALSE, allowParallel = TRUE)
z <- sort(c(a,b))
z <- z+1
z <- unique(z)

## Preprocessing data. Standalone: Transforms can be modeled from training data and
## applied to multiple datasets. The model of the transform is prepared using the 
## preProcess() function and applied to a dataset using the predict() function.

training <- training[,-z]
testing <- testing[,-z]
Group <- training$Grupo
training$Grupo <- NULL
Group_test <- testing$Grupo
testing$Grupo <- NULL

preProcValues <- preProcess(training, method = c("center", "scale"))

trainTransformed <- predict(preProcValues, training)
testTransformed <- predict(preProcValues, testing)

testTransformed <- droplevels(testTransformed)
trainTransformed <- droplevels(trainTransformed)

### KNN  MODEL ###
set.seed(123)

## K=2
knnK2_2classes <- knn(train=trainTransformed, test=testTransformed, cl=Group, k=2)

#Calculate the proportion of correct classification for k = 2
ACC.4 <- 100 * sum(Grupo_test == knnK2_2classes)/NROW(Group_test)
table(knnK2_2classes ,Group_test)
confusionMatrix(table(knnK2_2classes ,Group_test))

## K=3
knnK3_2classes <- knn(train=trainTransformed, test=testTransformed, cl=Group, k=3)

#Calculate the proportion of correct classification for k = 3
ACC.4 <- 100 * sum(Grupo_test == knnK3_2classes)/NROW(Group_test)
table(knnK3_2classes ,Group_test)
confusionMatrix(table(knnK3_2classes ,Group_test))

