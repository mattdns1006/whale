setwd("~/University/writeUps/15 Month Report/results/whale/ROC/")
hm = read.csv("hmROC.csv")[,2]
nhm = read.csv("nhmROC.csv")[,2]
df = data.frame(
  threshold = seq(0.1,0.9,length.out = 20),
  histMatch = hm,
  noHistMatch = nhm)
library(ggplot2)
library(reshape)

df = melt(df,id="threshold")

ggplot(aes(x=threshold,y=value,color=variable),data = df) + geom_line() + 
  scale_x_continuous(breaks = seq(0,1,by = 0.1)) + 
  ylab("Dice score")

#ggsave("diceROC.jpg")

