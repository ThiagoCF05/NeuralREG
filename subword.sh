# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=89500

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/home/tcastrof/workspace/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/tcastrof/workspace/nematus

# train BPE
cat data/train/refex.txt | $subword_nmt/learn_bpe.py -s $bpe_operations > data/models/subword.bpe

# apply BPE
$subword_nmt/apply_bpe.py -c data/models/subword.bpe < data/$prefix.tc.$SRC > data/$prefix.bpe.$SRC
for prefix in train dev eval1 test
 do
  $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$SRC > data/$prefix.bpe.$SRC
  $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$TRG > data/$prefix.bpe.$TRG
 done

# build network dictionary
$nematus/data/build_dictionary.py data/train.bpe.$SRC data/train.bpe.$TRG