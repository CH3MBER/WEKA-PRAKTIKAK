package holdOut;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.time.LocalDate;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Main {
	public static void main (String[] args) throws Exception {
		// DATA LORTU
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1) {data.setClassIndex(data.numAttributes() - 1);}
		
		// HOLD-OUT BANAKETA EGIN 
		data.randomize(new Random(42));
		RemovePercentage rp = new RemovePercentage();
		rp.setPercentage(66);
		rp.setInputFormat(data);
		Instances test = Filter.useFilter(data, rp);
		rp.setInvertSelection(true);
		rp.setInputFormat(data);
		Instances train = Filter.useFilter(data, rp);
		
		// TRAIN EBALUATU TEST ERABILIZ
		Classifier nb = new NaiveBayes();
		Evaluation eval = new Evaluation(data);
		nb.buildClassifier(train);
		eval.evaluateModel(nb, test);
		
		BufferedWriter writer = new BufferedWriter (new FileWriter(args[1]));
		writer.write("EXEKUZIO DATA: " + LocalDate.now() + "\n");
		writer.write("ARGUMENTUAK: " + args[0] + "  " + args[1] + "\n");
		
		// KLASE MINORITARIOA LORTU
		int[] KBalioak = data.attributeStats(data.classIndex()).nominalCounts;
		int KMinIndex = Utils.minIndex(KBalioak);
		String KMinIzena = data.classAttribute().value(KMinIndex);
		
		writer.write("KLASE MINORITARIOA (" + KMinIzena + "):\n");
		writer.write("     -PRECISION: " + eval.precision(KMinIndex) + "\n");
		writer.write("     -RECALL: " + eval.recall(KMinIndex) + "\n");
		writer.write("     -F-MEASURE: " + eval.fMeasure(KMinIndex) + "\n");
		writer.write("WEIGHTED PRECISION: " + eval.weightedPrecision() + "\n");
		writer.write("WEIGHTED RECALL: " + eval.weightedRecall() + "\n");
		writer.write("WEIGHTED F-MEASURE: " + eval.weightedFMeasure() + "\n\n");
		writer.write(eval.toMatrixString("===== NAHAZMEN MATRIZEA ====="));
		RepeatedHO.repeatedHO(data, args, writer);
		writer.flush();
		writer.close();
		System.out.println("\n\n\n\n (!!!) FITXATEGIA SORTUTA (!!!) \n\n\n\n");
	}
}
