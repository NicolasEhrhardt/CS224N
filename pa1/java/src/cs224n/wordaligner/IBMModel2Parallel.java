package cs224n.wordaligner;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.Counters;
import cs224n.util.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

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
public class IBMModel2Parallel implements WordAligner {

    private static final long serialVersionUID = 1315751943476440515L;

    // from the training data.
    private CounterMap<String, String> lexicalProb;
    private CounterMap<Pair<Integer, Pair<Integer, Integer>>, Integer> positionProb;

    public Alignment align(SentencePair sentencePair) {
        // Placeholder code below.
        // handout to predict alignments based on the counts you collected with train().
        Alignment alignment = new Alignment();

        List<String> sourceWords = sentencePair.getSourceWords();
        List<String> targetWords = sentencePair.getTargetWords();
        Pair<Integer, Integer> pairLength = new Pair<Integer, Integer>(sourceWords.size(), targetWords.size());

        CounterMap<String, String> alignProb = new CounterMap<String, String>();
        for (int i = 0; i < targetWords.size(); i++) {
            String targetWord = targetWords.get(i);
            Pair<Integer, Pair<Integer, Integer>> positionPair = new Pair<Integer, Pair<Integer, Integer>>(i, pairLength);
            for (int j = 0; j < sourceWords.size(); j++) {
                String sourceWord = sourceWords.get(j);
                alignProb.setCount(targetWord, sourceWord, lexicalProb.getCount(sourceWord, targetWord) * positionProb.getCount(positionPair, j));
            }
        }

        for (int tgtIndex = 0; tgtIndex < targetWords.size(); tgtIndex++) {
            String tgtWord = targetWords.get(tgtIndex);
            String bestSourceWord = alignProb.getCounter(tgtWord).argMax();
            int srcIndex = sourceWords.indexOf(bestSourceWord);
            alignment.addPredictedAlignment(tgtIndex, srcIndex);
        }

        return alignment;
    }

    public void train(List<SentencePair> trainingPairs) {
        System.out.println(String.format("%s, Init lexical prob from IBM1", this.getClass().getName()));
        IBMModel1Parallel IBM1 = new IBMModel1Parallel();
        IBM1.train(trainingPairs);
        lexicalProb = IBM1.getLexicalProb();

        System.out.println(String.format("%s, Init at random for position probabilities", this.getClass().getName()));
        positionProb = new CounterMap<Pair<Integer, Pair<Integer, Integer>>, Integer>();

        for(SentencePair pair : trainingPairs) {
            List<String> targetWords = pair.getTargetWords();
            List<String> sourceWords = new ArrayList<String>();
            sourceWords.addAll(pair.getSourceWords());
            Pair<Integer, Integer> pairLength = new Pair<Integer, Integer>(sourceWords.size(), targetWords.size());

            // Add one null word per sentence pair
            sourceWords.add(NULL_WORD);

            for (int i = 0; i < targetWords.size(); i++) {
                Pair<Integer, Pair<Integer, Integer>> positionPair = new Pair<Integer, Pair<Integer, Integer>>(i, pairLength);
                for (int j = 0; j < sourceWords.size(); j++) {
                    positionProb.setCount(positionPair, j, Math.random());
                }
            }
        }
        positionProb = Counters.conditionalNormalize(positionProb);

        System.out.println(String.format("%s, Training", this.getClass().getName()));
        for (int iter = 1; iter <= 50; iter++) {
            final CounterMap<String, String> countLexical = new CounterMap<String, String>();
            final CounterMap<Pair<Integer, Pair<Integer, Integer>>, Integer> countPosition = new CounterMap<Pair<Integer, Pair<Integer, Integer>>, Integer>();
            int njobs = 10;
            int sz = trainingPairs.size();
            int jobsz = (sz + 1) / njobs;

            ExecutorService executorService = Executors.newFixedThreadPool(njobs);
            for (int i = 0; i < njobs; i++) {
                final List<SentencePair> jobTrainingPairs = trainingPairs.subList(i * jobsz, Math.min((i + 1) * jobsz, sz));
                Runnable pairJob = new Runnable() {
                    public void run() {
                        for (final SentencePair pair : jobTrainingPairs) {

                            List<String> targetWords = pair.getTargetWords();
                            List<String> sourceWords = new ArrayList<String>();
                            sourceWords.addAll(pair.getSourceWords());

                            Pair<Integer, Integer> pairLength = new Pair<Integer, Integer>(sourceWords.size(), targetWords.size());

                            // Add one null word per sentence pair
                            sourceWords.add(NULL_WORD);

                            for (int i = 0; i < targetWords.size(); i++) {
                                String targetWord = targetWords.get(i);
                                Pair<Integer, Pair<Integer, Integer>> positionPair = new Pair<Integer, Pair<Integer, Integer>>(i, pairLength);

                                Counter<String> posterior = new Counter<String>();
                                double norm = 0.;

                                // computing posterior and total normalization
                                for (int j = 0; j < sourceWords.size(); j++) {
                                    String sourceWord = sourceWords.get(j);
                                    double sourceIncrement = lexicalProb.getCount(sourceWord, targetWord) * positionProb.getCount(positionPair, j);
                                    posterior.incrementCount(sourceWord, sourceIncrement);
                                    norm += sourceIncrement;
                                }

                                // updating counts
                                for (int j = 0; j < sourceWords.size(); j++) {
                                    String sourceWord = sourceWords.get(j);
                                    double sourceNormalizedIncrement = posterior.getCount(sourceWord) / norm;
                                    countLexical.incrementCount(sourceWord, targetWord, sourceNormalizedIncrement);
                                    countPosition.incrementCount(positionPair, j, sourceNormalizedIncrement);
                                }
                            }
                        }
                    }
                };
                executorService.submit(pairJob);
            }
            executorService.shutdown();

            try {
                executorService.awaitTermination(1, TimeUnit.MINUTES);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            CounterMap<String, String> oldLexicalProb = lexicalProb;
            CounterMap<Pair<Integer, Pair<Integer, Integer>>, Integer> oldPositionProb = positionProb;

            lexicalProb = Counters.conditionalNormalizeInPlace(countLexical);
            positionProb = Counters.conditionalNormalizeInPlace(countPosition);

            double epsLexicalProb = Counters.epsilon(oldLexicalProb, lexicalProb);
            double epsPositionProb = Counters.epsilon(oldPositionProb, positionProb);

            System.out.println(String.format(
                    "%s, Iteration %d, epsilon lexical %6.3e, epsilon position %6.3e",
                    this.getClass().getName(), iter, epsLexicalProb, epsPositionProb));
            if (epsLexicalProb < 2.e-6 && epsPositionProb < 2.e-6) {
                break;
            }
        }
    }


}
