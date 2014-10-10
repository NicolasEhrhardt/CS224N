package cs224n.wordaligner;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.Counters;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

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
public class IBMModel1Parallel implements WordAligner {

    private static final long serialVersionUID = 1315751943476440515L;

    // TODO: Use arrays or Counters for collecting sufficient statistics
    // from the training data.
    private  CounterMap<String, String> lexicalProb = new CounterMap<String, String>();

    public Alignment align(SentencePair sentencePair) {
        // Placeholder code below.
        // TODO Implement an inference algorithm for Eq.1 in the assignment
        // handout to predict alignments based on the counts you collected with train().
        Alignment alignment = new Alignment();

        List<String> sourceWords = sentencePair.getSourceWords();
        List<String> targetWords = sentencePair.getTargetWords();

        CounterMap<String, String> alignProb = new CounterMap<String, String>();
        for (String targetWord: targetWords) {
            for (String sourceWord: sourceWords) {
                alignProb.setCount(targetWord, sourceWord, lexicalProb.getCount(sourceWord, targetWord));
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
        System.out.println(String.format("%s, Init at uniform", this.getClass().getName()));
        lexicalProb = new CounterMap<String, String>();
        for(SentencePair pair : trainingPairs) {
            List<String> targetWords = pair.getTargetWords();
            List<String> sourceWords = new ArrayList<String>();
            sourceWords.addAll(pair.getSourceWords());

            // Add one null word per sentence pair
            sourceWords.add(NULL_WORD);
            for (String target : targetWords) {
                for (String source : sourceWords) {
                    lexicalProb.setCount(source, target, 1);
                }
            }
        }
        lexicalProb = Counters.conditionalNormalize(lexicalProb);

        System.out.println(String.format("%s, Training", this.getClass().getName()));
        for (int iter = 1; iter <= 50; iter++) {
            final CounterMap<String, String> countLexical = new CounterMap<String, String>();
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

                            // Add one null word per sentence pair
                            sourceWords.add(NULL_WORD);

                            for (String targetWord : targetWords) {
                                Counter<String> posterior = new Counter<String>();
                                double norm = 0.;

                                for (String sourceWord : sourceWords) {
                                    double sourceIncrement = lexicalProb.getCount(sourceWord, targetWord);
                                    posterior.incrementCount(sourceWord, sourceIncrement);
                                    norm += sourceIncrement;
                                }

                                for (String sourceWord : sourceWords) {
                                    countLexical.incrementCount(sourceWord, targetWord, posterior.getCount(sourceWord) / norm);
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

            lexicalProb = Counters.conditionalNormalizeInPlace(countLexical);

            double eps = Counters.epsilon(oldLexicalProb, lexicalProb);
            System.out.println(String.format("%s, Iteration %d, epsilon %6.3e", this.getClass().getName(), iter, eps));
             if (eps < 1.e-6) {
                break;
            }
        }
    }

    public CounterMap<String, String> getLexicalProb() {
        return lexicalProb;
    }
}
