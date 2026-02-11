package holdOut;

import java.io.BufferedWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.Utils;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class RepeatedHO {
	public static void repeatedHO(Instances data, String[] args, BufferedWriter writer) throws Exception {
		Stats stats = new Stats();
		
		for (int i = 0; i < 50; i++) {
			data.randomize(new Random());
			RemovePercentage rp = new RemovePercentage();
			rp.setPercentage(66);
			rp.setInputFormat(data);
			Instances test = Filter.useFilter(data, rp);
			rp.setInvertSelection(true);
			Instances train = Filter.useFilter(data, rp);
			Classifier nb = new NaiveBayes();
			Evaluation eval = new Evaluation(data);
			nb.buildClassifier(train);
			eval.evaluateModel(nb, test);
			int[] kBalioak = data.attributeStats(data.classIndex()).nominalCounts;
			int kMinIndex = Utils.minIndex(kBalioak);
			double unekoRecall = eval.recall(kMinIndex);
			stats.add(unekoRecall);
		}
		stats.calculateDerived();
		
		double batazB = stats.mean;
		double desb = stats.stdDev;
		
		writer.write("\n===== REPEATED HOLD-OUT =====");
		writer.write("\nBATAZBESTEKOA: " + batazB);
		writer.write("\nDESBIDERAPENA: " + desb);
	}
}
