WritehealthModel_interaction <- function(
    output.dir = paste0(getwd(), "/code/"),
    modelname = "health_model_interaction_randslope",
    PPC = FALSE,
    BDAprior = FALSE
){
  modelpath <- file.path(output.dir, paste0(modelname, ifelse(BDAprior, "_BDAprior", ""), ".txt"))
  
  cat("model {", file = modelpath, append = FALSE, fill = TRUE)
  
  # ---- 模型主体 ----
  cat("
  for (i in 1:n) {
    y.i[i] ~ dbin(p.i[i], 1)
    logit(p.i[i]) <-
      b.0
      + b.1[prov.i[i]] * health.i[i]
      + b.2 * log.gdp[i]  
      + b.3 * edu.i[i]
      + b.4 * hix.i[i]
      + b.5 * health.i[i] * log.gdp[i]  
      + v[prov.i[i]]
      + gamma[year.i[i]]

    loglike.i[i] <- logdensity.bin(y.i[i], p.i[i], 1)
    yrep.i[i] ~ dbin(p.i[i], 1)
  }
  ", file = modelpath, append = TRUE)
  
  # ---- 随机斜率项 ----
  cat("
  for (j in 1:J) {
    b.1[j] ~ dnorm(mu.b1, tau.b1)
  }
  mu.b1 ~ dnorm(0, 0.0001)
  tau.b1 <- pow(sigma.b1, -2)
  sigma.b1 ~ dunif(0, 10)
  ", file = modelpath, append = TRUE)
  
  # ---- 其他回归系数 ----
  if (BDAprior) {
    message("⚠️ BDAprior enabled. Ensure inputs are standardized!")
    cat("
    b.0 ~ dt(0, 1/2.5^2, 1)
    b.2 ~ dt(0, 1/2.5^2, 1)
    b.3 ~ dt(0, 1/2.5^2, 1)
    b.4 ~ dt(0, 1/2.5^2, 1)
    b.5 ~ dt(0, 1/2.5^2, 1)
    ", file = modelpath, append = TRUE)
  } else {
    cat("
    b.0 ~ dnorm(0, 0.0001)
    b.2 ~ dnorm(0, 0.0001)
    b.3 ~ dnorm(0, 0.0001)
    b.4 ~ dnorm(0, 0.0001)
    b.5 ~ dnorm(0, 0.0001)
    ", file = modelpath, append = TRUE)
  }
  
  # ---- 省份随机截距 ----
  cat("
  for (j in 1:J) {
    v[j] ~ dnorm(0, tau.v)
  }
  tau.v <- pow(sigma.v, -2)
  sigma.v ~ dunif(0, 10)
  ", file = modelpath, append = TRUE)
  
  # ---- 年份固定效应 ----
  cat("
  for (t in 1:T) {
    gamma[t] ~ dnorm(0, tau.gamma)
  }
  tau.gamma <- pow(sigma.gamma, -2)
  sigma.gamma ~ dunif(0, 10)
  ", file = modelpath, append = TRUE)
  
  # ---- 后验预测派生量 ----
  if (PPC){
    message("✅ PPC enabled: computing tyrep")
    cat("
    tyrep <- sum(yrep.i[] * (log.gdp[] < gdpcutoff)) / sum(log.gdp[] < gdpcutoff) 
    ", file = modelpath, append = TRUE)
  }
  
  cat("}  # end model", file = modelpath, append = TRUE, fill = TRUE)
  
  return(modelpath)
}
