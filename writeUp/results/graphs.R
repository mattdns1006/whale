setwd("~/University/writeUps/15 Month Report/results/whale/")
rnames <- function(df){
  return(cbind(x = rownames(df), df))
}
library(ggplot2)
library(reshape)

for(i in c(1,3)){

#hmTr = read.csv("hist/trainPerf.csv")[,i]
#nhmTr = read.csv("noHist/trainPerf.csv")[,i]
hmTe = read.csv("hist/testPerf.csv")[,i]
nhmTe = read.csv("noHist/testPerf.csv")[,i]
nEpochs = length(hmTr)

if(i == 1){
  name = "MSE"
}else{
  name = "Dice score"
}

df = data.frame(
  trainOrTest = c("",""),
  histMatch = c("Histogram match","Normal"),
  t(cbind(hmTe,nhmTe))
)
df = melt(df,id.vars = c("trainOrTest","histMatch"))
colnames(df)[colnames(df)=="value"] = name
df$epochs = sort(rep(c(1:nEpochs),2))
p <- ggplot(data=df,aes(x=epochs,y=df[colnames(df)==name],color = interaction(trainOrTest,histMatch,sep=" "),
                        group=interaction(trainOrTest,histMatch))) +
  geom_point() +
  geom_smooth(span = 0.1) +
  
  xlab('Epoch number') + 
  scale_x_continuous(breaks = seq(0,nEpochs,by = 2)) + 
  ylab(name) + 
  theme(legend.position = "bottom") +
  labs(colour = "") 
print(p)
ggsave(paste(name,".jpg",sep = ""))
}

