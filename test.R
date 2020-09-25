library("devtools")
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
devtools::load_all(paste(getwd(), "/MLDS", sep=""))

mydata1 = read.table("data.txt", sep="\t")
x1.mlds<-mlds(mydata1)
print(summary(x1.mlds))