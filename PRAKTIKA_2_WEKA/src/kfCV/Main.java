package kfCV;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.time.LocalDate;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
	public static void main (String[] args) throws Exception {
		// ARGUMENTUAK EZ BADIRA IDAZTEN
		try {
			args[1].chars();
		} catch (Exception e) {
			System.out.println("ERROREA! Programa honetan 2 argumentu bidali behar dira.");
			System.exit(0);
		}
		
		// DATUAK LORTU
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if(data.classIndex() == -1) {data.setClassIndex(data.numAttributes() - 1);}
				
		// K-FOLD CROSS VALIDATION EGIN
		Evaluation eval = new Evaluation(data);
		Classifier nb = new NaiveBayes();
		eval.crossValidateModel(nb, data, 5, new Random(42));
		
		BufferedWriter writer = new BufferedWriter(new FileWriter(args[1]));
		writer.write(eval.toMatrixString());
		writer.write("\n");
		for (int i = 0; i < data.classAttribute().numValues(); i++) {
			writer.write("Precision-" + i + ": " + eval.precision(i) + "\n");
		}
		writer.write("Weighted Avg: " + eval.weightedPrecision() + "\n");
		writer.write("Exekuzio data: " + LocalDate.now() + "\n");
		writer.write("Argumentuak: " + args[0] + " " +args[1] + "\n\n");
		writer.write("Emaitzak:");
		writer.write(eval.toSummaryString());
		writer.flush();
		writer.close();
		
		System.out.println("\n\n\n\n(!!!) FITXATEGIA SORTUTA (!!!)\n\n\n\n");
	}
}
