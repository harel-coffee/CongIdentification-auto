redundant <- function(df){
  data=df
  redundant_variables<-redun(~.,data=data, nk = 0)
  #print(redundant_variables)
  return (redundant_variables)
}



