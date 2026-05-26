library(dplyr)
library(rjags)
library(R2jags)
library(loo) 
library(tidyverse)
#use your data file to revise the address 
setwd("C:\\Users\\Derick\\Desktop\\计算社会科学课程学习\\5210 advanced topics in  statistics\\final project\\data")
# 读取 .dta 文件
load("merged_data_clean.Rdata")
health <- merged_data_clean
health$pergdp <- log(health$pergdp)
colnames(health)[colnames(health) == "pergdp"] <- "log.gdp"

figdir <- "fig/"
# automatically creates a subfolder under the working directory
dir.create(figdir, showWarnings = FALSE)
set.seed(1234)  # 设置随机种子以保证可重复性
n <- nrow(health)
test_index <- sample(1:n, size = floor(0.2 * n))  # 20% 
train_index <- setdiff(1:n, test_index)

# 构造训练集和测试集数据
train_data <- health[train_index, ]
test_data <- health[test_index, ]
n <- nrow(train_data)
# 提取训练数据的建模变量并中心化
y.train <- train_data$chronic_disease
ratio.train <- train_data$healthcare_ratio
loggdp.train <- train_data$log.gdp
edu.train <- train_data$education
index.train <- train_data$health_index

c.ratio.train <- ratio.train - mean(ratio.train)
c.loggdp.train <- loggdp.train - mean(loggdp.train)
c.edu.train <- edu.train - mean(edu.train)
c.index.train <- index.train - mean(index.train)

# 创建省份索引和年份索引（注意 factorize 后加 1，因 JAGS 从 1 开始）
train_data$prov.i <- as.numeric(factor(train_data$province))
train_data$year.i <- as.numeric(factor(train_data$year))

J <- length(unique(train_data$prov.i))
T <- length(unique(train_data$year.i))

source("C:\\Users\\Derick\\Desktop\\计算社会科学课程学习\\5210 advanced topics in  statistics\\final project\\data\\write_health_model.R")
modelfilepath1 <- WritehealthModel_interaction(
  modelname = "health_model_with_interaction",
  PPC = TRUE,
)

source("C:\\Users\\Derick\\Desktop\\计算社会科学课程学习\\5210 advanced topics in  statistics\\final project\\data\\write_health_model.R")
# example
# 模型 1：含交互项（默认 addinteraction = TRUE）
modelfilepath1 <- WritehealthModel_interaction(
  modelname = "health_model_with_interaction",
  PPC = TRUE,
)
jagsdata <- list(
  n = n,
  y.i = y.train,
  health.i = c.ratio.train,
  log.gdp = c.loggdp.train,
  edu.i = c.edu.train,
  hix.i = c.index.train,
  gdpcutoff = quantile(c.loggdp.train, 0.25),
  prov.i=train_data$prov.i,
  year.i=train_data$year.i,
  J=J,
  T=T
)
params <- c(
  "b.0", "b.1", "b.2", "b.3", "b.4", "b.5",  # 回归系数
  "v", "sigma.v",                          # 省份效应
  "gamma", "sigma.gamma",                 # 年效应
  "p.i", "yrep.i", "tyrep", "loglike.i"   # 模型预测 + 检查
)
# 含交互项模型
mod1 <- jags.parallel(jagsdata,
                      parameters.to.save = params,
                      model.file = modelfilepath1,
                      n.chains = 3, n.burnin = 300, n.iter = 600, n.thin = 1)
mcmc.array <- mod1$BUGSoutput$sims.array
# quick check for convergence (for full analysis,
# more comprehensive checks would be preferred)
summary(mod1$BUGSoutput$summary[, c("n.eff", "Rhat")])

# ========== 后验预测检查部分 (PPC for Bernoulli model) ==========


# 所有回归系数
mod1$BUGSoutput$summary[grep("^b\\.", rownames(mod1$BUGSoutput$summary)), c("mean", "2.5%", "97.5%")]



# ========= 1. 提取 p.i[i] 后验概率和 CI =========
p.names <- grep("^p\\.i\\[", dimnames(mcmc.array)[[3]], value = TRUE)
p.array <- mcmc.array[, , p.names]
S <- dim(p.array)[1] * dim(p.array)[2]
n.obs <- length(p.names)
p.matrix <- matrix(p.array, nrow = S, ncol = n.obs)

p.mean  <- colMeans(p.matrix)
p.lower <- apply(p.matrix, 2, quantile, probs = 0.025)
p.upper <- apply(p.matrix, 2, quantile, probs = 0.975)

# ========= 2. 提取 yrep.i[i] 后验预测分布 =========
yrep.names <- grep("^yrep\\.i\\[", dimnames(mcmc.array)[[3]], value = TRUE)
yrep.array <- mcmc.array[, , yrep.names]
yrep.matrix <- matrix(yrep.array, nrow = S, ncol = n.obs)

# ========= 3. PIT Histogram =========
pit.values <- ifelse(y.train == 1, p.mean, 1 - p.mean)
pdf(paste0(figdir, "PPC_PIT_histogram.pdf"), width = 6, height = 5, family = "Times")
hist(pit.values,
     breaks = 10, main = "PIT Histogram", xlab = "PIT value",
     col = "skyblue", border = "white", freq = FALSE)
curve(dunif(x), add = TRUE, col = "lightpink", lwd = 2, lty = 2)
dev.off()

# ========= 4. Proportion of 1s: Observed vs Replicated =========
yrep.prop.1 <- rowMeans(yrep.matrix)
obs.prop.1 <- mean(y.train)

pdf(paste0(figdir, "PPC_prop1_hist.pdf"), width = 6, height = 5, family = "Times")
hist(yrep.prop.1, freq = FALSE,
     main = "Proportion of y = 1 in Posterior Predictive Draws",
     xlab = "Proportion of 1s", col = "gray80", border = "white")
abline(v = obs.prop.1, col = "lightblue", lwd = 2)
legend("topright", legend = "Observed", col = "red", lty = 1, cex = 1.2)
dev.off()

# ========= 5. Coverage =========
coverage <- mean(y.train >= p.lower & y.train <= p.upper)
cat("Posterior predictive coverage (95% CI):", coverage, "\n")

# ========= 6. 可选：观测 vs p.i 可视化（如果你有 year 或 id） =========
# 如果你有变量 train_data$year，可使用：
if ("year" %in% colnames(train_data)) {
  year.vec <- train_data$year
  o <- order(year.vec)
  pdf(paste0(figdir, "PPC_p_mean_vs_obs.pdf"), width = 9, height = 5, family = "Times")
  plot(y.train[o] ~ year.vec[o], pch = 1, col = "black", ylim = c(0, 1.05),
       xlab = "Year", ylab = "Observed / Predicted")
  lines(p.mean[o] ~ year.vec[o], col = "purple", lwd = 2)
  lines(p.lower[o] ~ year.vec[o], col = "green3", lty = 2)
  lines(p.upper[o] ~ year.vec[o], col = "green3", lty = 2)
  legend("topright", legend = c("Observed", "Predicted mean", "95% CI"),
         col = c("black", "purple", "green3"), lty = c(NA, 1, 2),
         pch = c(1, NA, NA), lwd = 2, cex = 1.2)
  dev.off()
}


# 提取测试集变量
y.test <- test_data$chronic_disease
ratio.test <- test_data$healthcare_ratio
loggdp.test <- test_data$log.gdp
edu.test <- test_data$education
index.test <- test_data$health_index

# 用训练集均值进行中心化
c.ratio.test <- ratio.test - mean(ratio.train)
c.loggdp.test <- loggdp.test - mean(loggdp.train)
c.edu.test <- edu.test - mean(edu.train)
c.index.test <- index.test - mean(index.train)

# 设置省份和年份的索引，要确保和训练集的 levels 一致
test_data$prov.i <- as.numeric(factor(test_data$province, levels = levels(factor(train_data$province))))
test_data$year.i <- as.numeric(factor(test_data$year, levels = levels(factor(train_data$year))))

n.test <- nrow(test_data)
S <- dim(mod1$BUGSoutput$sims.list$b.0)[1]  # 后验样本数量
# 从模型中提取参数后验样本
b0 <- mod1$BUGSoutput$sims.list$b.0
b1 <- mod1$BUGSoutput$sims.list$b.1
b2 <- mod1$BUGSoutput$sims.list$b.2
b3 <- mod1$BUGSoutput$sims.list$b.3
b4 <- mod1$BUGSoutput$sims.list$b.4
b5 <- mod1$BUGSoutput$sims.list$b.5
v <- mod1$BUGSoutput$sims.list$v
gamma <- mod1$BUGSoutput$sims.list$gamma

# 构建预测矩阵（S x n.test）
p.test <- matrix(NA, nrow = S, ncol = n.test)

for (s in 1:S) {
  eta <- b0[s] +
    b1[s] * c.ratio.test +
    b2[s] * c.loggdp.test +
    b3[s] * c.edu.test +
    b4[s] * c.index.test +
    b5[s] * c.loggdp.test * c.ratio.test +  # 如果模型中含交互项
    v[s, test_data$prov.i] +
    gamma[s, test_data$year.i]
  
  p.test[s, ] <- 1 / (1 + exp(-eta))
}
# 预测概率的均值
p.mean <- colMeans(p.test)

# 误差度量
RMSE <- sqrt(mean((y.test - p.mean)^2))
MAE <- mean(abs(y.test - p.mean))

cat("RMSE:", RMSE, "\n")
cat("MAE:", MAE, "\n")

# ✅ 正确 PIT 值计算（对于 Bernoulli 数据）
pit_values <- ifelse(test_data$chronic_disease == 1,
                     colMeans(p.test),
                     1 - colMeans(p.test))

# ✅ 保存 PIT histogram 图
pdf("fig/pit_histogram.pdf")
hist(pit_values,
     breaks = 10,
     main = "PIT Histogram",
     xlab = "PIT Value",
     col = "skyblue",
     border = "white",
     freq = FALSE)
curve(dunif(x), add = TRUE, col = "red", lwd = 2, lty = 2)  # 添加 uniform 参考线
dev.off()

# ✅ 模拟 posterior predictive draws（Bernoulli samples）
# yrep: posterior predictive draws (0 or 1) from p.test
y.rep <- matrix(rbinom(n.test * S, size = 1, prob = c(p.test)), nrow = S)

# ✅ 检查有多少次预测复现了真实 y 值
pointwise_coverage <- sapply(1:n.test, function(i) {
  mean(y.rep[, i] == y.test[i])
})


pred_intervals <- apply(p.test, 2, quantile, probs = c(0.025, 0.975))
coverage <- mean(test_data$chronic_disease >= pred_intervals[1, ] & 
                   test_data$chronic_disease <= pred_intervals[2, ])
cat("95% prediction interval coverage:", coverage, "\n")

library(loo)

# 提取 log-likelihood（你已经保存了 loglike.i[i]）
loglike.names <- grep("^loglike\\.i\\[", dimnames(mcmc.array)[[3]], value = TRUE)

# 构建 S x n 的 log-likelihood 矩阵
loglike.matrix <- do.call(cbind, lapply(loglike.names, function(n) {
  matrix(mcmc.array[, , n], ncol = 1)
}))

# 计算 LOO 和 WAIC
loo.out <- loo(loglike.matrix)
waic.out <- waic(loglike.matrix)

print(loo.out)
print(waic.out)

boxplot(p.mean ~ y.test,
        names = c("0(No chronic disease)", "1(Chronic disease)"),
        ylab = "Predicted Probability",
        xlab = "Observed Label",
        col = c("skyblue", "tomato"))

# ========== 7. Residual Error Plot ==========
residuals <- y.test - p.mean

pdf("fig/residuals_plot.pdf", width = 6, height = 5, family = "Times")
plot(p.mean, residuals,
     xlab = "Predicted Probability",
     ylab = "Residual (Observed - Predicted)",
     main = "Residual Error Plot",
     pch = 16, col = rgb(0, 0, 1, 0.3))
abline(h = 0, col = "red", lty = 2, lwd = 2)
dev.off()



