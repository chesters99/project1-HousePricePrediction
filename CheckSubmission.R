options(digits=5)
test <- read.csv("../Ames_data.csv")
test.y = test[,c("PID","Sale_Price")]

pred <- read.csv("mysubmission3.txt")
names(test.y)[2] <- "True_Sale_Price"
pred <- merge(pred, test.y, by="PID")
sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))