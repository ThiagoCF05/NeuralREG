./multeval.sh eval --refs ../../stats/refs.txt \
                   --hyps-baseline ../../stats/only.txt \
                   --hyps-sys1 ../../stats/ferreira.txt \
                   --hyps-sys2 ../../stats/seq2seq.txt \
                   --hyps-sys3 ../../stats/catt.txt \
                   --hyps-sys4 ../../stats/hieratt.txt \
                   --meteor.language en

./multeval.sh eval --refs ../../stats/refs.txt \
                   --hyps-baseline ../../stats/ferreira.txt \
                   --hyps-sys1 ../../stats/seq2seq.txt \
                   --hyps-sys2 ../../stats/catt.txt \
                   --hyps-sys3 ../../stats/hieratt.txt \
                   --meteor.language en

./multeval.sh eval --refs ../../stats/refs.txt \
                   --hyps-baseline ../../stats/seq2seq.txt \
                   --hyps-sys1 ../../stats/catt.txt \
                   --hyps-sys2 ../../stats/hieratt.txt \
                   --meteor.language en

./multeval.sh eval --refs ../../stats/refs.txt \
                   --hyps-baseline ../../stats/catt.txt \
                   --hyps-sys1 ../../stats/hieratt.txt \
                   --meteor.language en