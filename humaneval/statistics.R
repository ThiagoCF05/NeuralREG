# == Data
datav <- read.csv(header=T, row.names=NULL, sep = ";", "/Users/thiagocastroferreira/Documents/Doutorado/NeuralREG/humaneval/official_results.csv")

# Fluency
print("Fluency")
# Friedman Test
fluency <- matrix(c(datav$original_fluency, datav$only_fluency, datav$ferreira_fluency, datav$seq2seq_fluency, datav$catt_fluency, datav$hier_fluency), nrow = nrow(datav))
friedman.test(fluency)

# Post-hoc: Wilcoxon
install.packages("reshape2")
library("reshape2")
fluency <- data.frame(matrix(c(datav$resp, datav$original_fluency, datav$only_fluency, datav$ferreira_fluency, datav$seq2seq_fluency, datav$catt_fluency, datav$hier_fluency), 
               nrow = nrow(datav)))
colnames(fluency) <- c("fluency/resp", "original", "only", "ferreira", "seq2seq", "catt", "hieratt")
datav.long <- melt(fluency, id.vars=c("fluency/resp"))

summary(fluency)

library("stats")
pairwise.wilcox.test(datav.long$value, datav.long$variable, paired = TRUE, p.adjust.method = "bonferroni")

# Grammar
print("Grammaticality")
# Friedman Test
grammar <- matrix(c(datav$original_grammar, datav$only_grammar, datav$ferreira_grammar, datav$seq2seq_grammar, datav$catt_grammar, datav$hier_grammar), nrow = nrow(datav))
friedman.test(grammar)

# Post-hoc: Wilcoxon
grammar <- data.frame(matrix(c(datav$resp, datav$original_grammar, datav$only_grammar, datav$ferreira_grammar, datav$seq2seq_grammar, datav$catt_grammar, datav$hier_grammar), 
                             nrow = nrow(datav)))
colnames(grammar) <- c("grammar/resp", "original", "only", "ferreira", "seq2seq", "catt", "hieratt")
datav.long <- melt(grammar, id.vars=c("grammar/resp"))

summary(grammar)

library("stats")
pairwise.wilcox.test(datav.long$value, datav.long$variable, paired = TRUE, p.adjust.method = "bonferroni")

# Clarity
print("Clarity")
# Friedman Test
clarity <- matrix(c(datav$original_clarity, datav$only_clarity, datav$ferreira_clarity, datav$seq2seq_clarity, datav$catt_clarity, datav$hier_clarity), nrow = nrow(datav))
friedman.test(clarity)

# Post-hoc: Wilcoxon
clarity <- data.frame(matrix(c(datav$resp, datav$original_clarity, datav$only_clarity, datav$ferreira_clarity, datav$seq2seq_clarity, datav$catt_clarity, datav$hier_clarity), 
                             nrow = nrow(datav)))
colnames(clarity) <- c("clarity/resp", "original", "only", "ferreira", "seq2seq", "catt", "hieratt")
datav.long <- melt(clarity, id.vars=c("clarity/resp"))

summary(clarity)

library("stats")
pairwise.wilcox.test(datav.long$value, datav.long$variable, paired = TRUE, p.adjust.method = "bonferroni")

