setwd("~/kaggle/whale/writeUp/results/")
rnames <- function(df){
return(cbind(x = rownames(df), df))
}
library(ggplot2)
library(reshape2)

i = 1
#hmTr = read.csv("hist/trainPerf.csv")[,i]
#nhmTr = read.csv("noHist/trainPerf.csv")[,i]
hmTe = read.csv("hist/testPerf.csv")[,i]
nhmTe = read.csv("noHist/testPerf.csv")[,i]
nEpochs = length(hmTe)

df = data.frame(
  trainOrTest = c("",""),
  histMatch = c("Histogram match","Normal"),
  t(cbind(hmTe,nhmTe))
)
name = "MSE"
df = melt(df,id.vars = c("trainOrTest","histMatch"))
colnames(df)[colnames(df)=="value"] = name
df$epochs = sort(rep(c(1:nEpochs),2))
attach(df)
y = `MSE`
#y = `Dice score`

p <- ggplot(data=df,aes(x=epochs,y=y,color = interaction(trainOrTest,histMatch,sep=" "),
                        group=interaction(trainOrTest,histMatch))) +
  geom_point() +
  geom_smooth(span = 0.1) +
  
  xlab('Epoch number') + 
  scale_x_continuous(breaks = seq(0,nEpochs,by = 2)) + 
  ylab(name) + 
  theme(legend.position = "bottom") +
  labs(colour = "") +
  guides(colour = guide_legend(override.aes = list(size=4,linetype=0))) + theme(legend.text=element_text(size=13)) +
  scale_color_manual(labels = c("Histogram matched", "Normal"), values = c("orange", "red")) 
print(p)
ggsave(paste("mse.jpg",sep = ""))

p
