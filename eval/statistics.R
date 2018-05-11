# == String Edit Distance
datav <- read.csv(header=T, row.names=NULL, sep = ";", "/Users/thiagocastroferreira/Documents/Doutorado/NeuralREG/eval/stats/r_distances.csv")

# Friedman Test
data <- matrix(c(datav$only, datav$ferreira, datav$seq2seq, datav$catt, datav$hieratt), nrow = nrow(datav))
friedman.test(data)

# Post-hoc: Wilcoxon
install.packages("reshape2")
library("reshape2")
datav.long <- melt(datav, id.vars=c("resp"))

library("stats")
pairwise.wilcox.test(datav.long$value, datav.long$variable, paired = TRUE, p.adjust.method = "bonferroni")

# == Referential Accuracies
datav <- read.csv(header=T, row.names=NULL, sep = ";", "/Users/thiagocastroferreira/Documents/Doutorado/NeuralREG/eval/stats/r_ref_acc.csv")

# bonferroni adjustment
k <- choose(dim(datav)[2]-1, 2)
mcnemar.test(datav$only, datav$ferreira, correct = FALSE)$p.value*k
mcnemar.test(datav$only, datav$seq2seq, correct = FALSE)$p.value*k
mcnemar.test(datav$only, datav$catt, correct = FALSE)$p.value*k
mcnemar.test(datav$only, datav$hieratt, correct = FALSE)$p.value*k

mcnemar.test(datav$ferreira, datav$seq2seq, correct = FALSE)$p.value*k
mcnemar.test(datav$ferreira, datav$catt, correct = FALSE)$p.value*k
mcnemar.test(datav$ferreira, datav$hieratt, correct = FALSE)$p.value*k

mcnemar.test(datav$seq2seq, datav$catt, correct = FALSE)$p.value*k
mcnemar.test(datav$seq2seq, datav$hieratt, correct = FALSE)$p.value*k

mcnemar.test(datav$catt, datav$hieratt, correct = FALSE)$p.value*k


# == Pronominal Accuracies
datav <- read.csv(header=T, row.names=NULL, sep = ";", "/Users/thiagocastroferreira/Documents/Doutorado/NeuralREG/eval/stats/r_pron_acc.csv")

# bonferroni adjustment
k <- choose(dim(datav)[2]-2, 2)
mcnemar.test(datav$ferreira, datav$seq2seq, correct = FALSE)$p.value*k
mcnemar.test(datav$ferreira, datav$catt, correct = FALSE)$p.value*k
mcnemar.test(datav$ferreira, datav$hieratt, correct = FALSE)$p.value*k

mcnemar.test(datav$seq2seq, datav$catt, correct = FALSE)$p.value*k
mcnemar.test(datav$seq2seq, datav$hieratt, correct = FALSE)$p.value*k

mcnemar.test(datav$catt, datav$hieratt, correct = FALSE)$p.value*k

# == Textual Accuracies
datav <- read.csv(header=T, row.names=NULL, sep = ";", "/Users/thiagocastroferreira/Documents/Doutorado/NeuralREG/eval/stats/r_text_acc.csv")

# bonferroni adjustment
k <- choose(dim(datav)[2]-1, 2)
mcnemar.test(datav$only, datav$ferreira, correct = FALSE)$p.value*k
mcnemar.test(datav$only, datav$seq2seq, correct = FALSE)$p.value*k
mcnemar.test(datav$only, datav$catt, correct = FALSE)$p.value*k
mcnemar.test(datav$only, datav$hieratt, correct = FALSE)$p.value*k

mcnemar.test(datav$ferreira, datav$seq2seq, correct = FALSE)$p.value*k
mcnemar.test(datav$ferreira, datav$catt, correct = FALSE)$p.value*k
mcnemar.test(datav$ferreira, datav$hieratt, correct = FALSE)$p.value*k

mcnemar.test(datav$seq2seq, datav$catt, correct = FALSE)$p.value*k
mcnemar.test(datav$seq2seq, datav$hieratt, correct = FALSE)$p.value*k

mcnemar.test(datav$catt, datav$hieratt, correct = FALSE)$p.value*k

