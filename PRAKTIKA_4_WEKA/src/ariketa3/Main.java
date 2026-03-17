package ariketa3;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

public class Main {
	public static void main(String[] args) throws Exception{
		// DATUAK LORTU
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if (data.classIndex()==-1){data.setClassIndex(data.numAttributes()-1);}
		
		int maxDOnena = -1;
		int numFOnena = -1;
		double baloreOnena = 0;
		
		System.out.println("Parametro ekorketa egiten...\n\n");
		
		// STRATIFIED HOLD OUT BANAKETA
		StratifiedRemoveFolds foldFilter = new StratifiedRemoveFolds();
		foldFilter.setNumFolds(5);      
		foldFilter.setFold(1);
		foldFilter.setInvertSelection(false); 
		foldFilter.setSeed(77);
		foldFilter.setInputFormat(data);
		Instances train = Filter.useFilter(data, foldFilter);

		// TEST: el fold 1 (20%)
		foldFilter.setInvertSelection(true); 
		foldFilter.setInputFormat(data);
		Instances test = Filter.useFilter(data, foldFilter);
		
		// KLASE MINORITARIOA LORTU
		AttributeStats klaseStats = data.attributeStats(data.classIndex());
		int minIndex = -1;
		int minBalorea = Integer.MAX_VALUE;
		
		for (int i = 0; i < klaseStats.nominalCounts.length; i++) {
			if (klaseStats.nominalCounts[i] < minBalorea && klaseStats.nominalCounts[i] != 0) {
				minBalorea = klaseStats.nominalCounts[i];
				minIndex = i;
			}
		}
		
		// PARAMETRO EKORKETA
		for (int i = 1; i <= 10; i++) {
			for (int x = 1; x <= 10; x++) {
				RandomForest rf = new RandomForest();
				rf.setMaxDepth(i);
				rf.setNumFeatures(x);
				rf.buildClassifier(train);
				Evaluation eval = new Evaluation(train);
				eval.evaluateModel(rf, test);

				if (eval.fMeasure(minIndex) > baloreOnena) {
					baloreOnena = eval.fMeasure(minIndex);
					maxDOnena = i;
					numFOnena = x;
				}
			}
		}
		System.out.println("maxDepth: " + maxDOnena);
		System.out.println("NumFeatures: " + numFOnena);
	}
}
