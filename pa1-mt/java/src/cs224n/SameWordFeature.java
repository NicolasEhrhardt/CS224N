package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;
import java.util.HashMap;

import edu.stanford.nlp.mt.util.FeatureValue;
import edu.stanford.nlp.mt.util.Featurizable;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.decoder.feat.RuleFeaturizer;
import edu.stanford.nlp.util.Generics;

/**
 * A rule featurizer.
 */
public class SameWordFeature implements RuleFeaturizer<IString, String> {

    @Override
    public void initialize() {
        // Do any setup here.
    }

    @Override
    public List<FeatureValue<String>> ruleFeaturize(
            Featurizable<IString, String> f) {

       // TODO: Return a list of features for the rule. Replace these lines
       // with your own feature.
       List<FeatureValue<String>> features = Generics.newLinkedList();

       // ON - WE
       boolean on = false;
        for (IString word: f.sourcePhrase) {
            if (word.toString().equals("on")) {
                on = true;
                break;
            }
        }

        boolean we = false;
        for (IString word: f.targetPhrase) {
            if (word.toString().equals("we")) {
                we = true;
                break;
            }
        }
        if (we && on) {
            features.add(new FeatureValue<String>("ONWE", 1.0));
        }

        // SAME WORD
        HashMap<Integer, Integer> sameWords = new HashMap<Integer, Integer>();
        for (IString tgtword: f.targetPhrase) {
            for (IString srcword: f.sourcePhrase) {
                if (tgtword.toString().equals(srcword.toString())) {
                    if (!sameWords.containsKey(srcword)) {
                        sameWords.put(srcword.getId(), 1);
                    } else {
                        sameWords.put(srcword.getId(), sameWords.get(srcword.getId()) + 1);
                    }
                }
            }
        }

        for (Integer value: sameWords.values()) {
            features.add(new FeatureValue<String>(String.format("FeatureSameWord:%d", value), 1.0));
        }

        return features;
    }

    @Override
    public boolean isolationScoreOnly() {
        return false;
    }
}
