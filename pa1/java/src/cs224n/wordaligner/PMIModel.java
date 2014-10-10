package cs224n.wordaligner;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.Counters;

import java.util.List;

/**
 * Simple word alignment baseline model that maps source positions to target 
 * positions along the diagonal of the alignment grid.
 *
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 *
 * @author Dan Klein
 * @author Spence Green
 */
public class PMIModel implements WordAligner {

    private static final long serialVersionUID = 1315751943476440515L;

    // TODO: Use arrays or Counters for collecting sufficient statistics
    // from the training data.
    private CounterMap<String, String> PMI;

    public Alignment align(SentencePair sentencePair) {
        // Placeholder code below.
        // TODO Implement an inference algorithm for Eq.1 in the assignment
        // handout to predict alignments based on the counts you collected with train().
        Alignment alignment = new Alignment();

        List<String> sourceWords = sentencePair.getSourceWords();
        List<String> targetWords = sentencePair.getTargetWords();

        CounterMap<String, String> localPMI = new CounterMap<String, String>();
        for (String targetWord: targetWords) {
            for (String sourceWord: sourceWords) {
                localPMI.setCount(targetWord, sourceWord, PMI.getCount(targetWord, sourceWord));
            }
        }

        for (int tgtIndex = 0; tgtIndex < targetWords.size(); tgtIndex++) {
            String tgtWord = targetWords.get(tgtIndex);
            String bestSourceWord = localPMI.getCounter(tgtWord).argMax();
            int srcIndex = sourceWords.indexOf(bestSourceWord);
            alignment.addPredictedAlignment(tgtIndex, srcIndex);
        }

        return alignment;
    }

    public void train(List<SentencePair> trainingPairs) {
        CounterMap<String, String> targetSourceCounts = new CounterMap<String,String>();
        Counter<String> sourceCounts = new Counter<String>();
        Counter<String> targetCounts = new Counter<String>();

        System.out.println("Get counts");
        for(SentencePair pair : trainingPairs){
            List<String> targetWords = pair.getTargetWords();
            List<String> sourceWords = pair.getSourceWords();

            // Add one null word per sentence pair
            sourceWords.add(NULL_WORD);

            // Get global frequencies
            for (String source: sourceWords) {
                sourceCounts.incrementCount(source, 1.0);
            }
            for (String target: targetWords) {
                targetCounts.incrementCount(target, 1.0);
            }

            for(String source: sourceWords){
                for(String target: targetWords){
                    targetSourceCounts.incrementCount(target, source, 1.0);
                }
            }
        }

        // normalize
        Counter<String> sourceProb = Counters.normalize(sourceCounts);
        Counter<String> targetProb = Counters.normalize(targetCounts);
        double nPairs = targetSourceCounts.totalSize();

        // compute PMI
        PMI = new CounterMap<String, String>();

        System.out.println("Compute PMI");
        for (String target: targetSourceCounts.keySet()) {
            Counter<String> targetCounter = targetSourceCounts.getCounter(target);
            for (String source: targetCounter.keySet()) {
                double sourceTargetCount = targetCounter.getCount(source);
                double srcProb = sourceProb.getCount(source);
                double tgtProb = targetProb.getCount(target);
                PMI.setCount(target, source, (sourceTargetCount / nPairs) / (srcProb * tgtProb));
            }
        }
    }
}
