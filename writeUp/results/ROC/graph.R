library(ggplot2)
library(reshape)
setwd("~/kaggle/whale/writeUp/results/ROC/")
hm = read.csv("hmROC.csv")[,2]
nhm = read.csv("nhmROC.csv")[,2]
df = data.frame(
  threshold = seq(0.1,0.9,length.out = 20),
  histMatch = hm,
  noHistMatch = nhm)

df = melt(df,id="threshold")
attach(df)

p = ggplot(aes(x=threshold,y=value,color=variable),data = df) + 
  scale_x_continuous(breaks = seq(0,1,by = 0.1)) + 
  geom_point() +
  geom_smooth(span = 0.1) +
  ylab("Dice score") + theme(legend.position = "bottom") +
  labs(colour = "") +
  guides(colour = guide_legend(override.aes = list(size=4,linetype=0))) + theme(legend.text=element_text(size=13)) +
  scale_color_manual(labels = c("Histogram matched", "Normal"), values = c("orange", "red")) 
print(p)
ggsave("dice.jpg")
