package ereduaSortu;

import java.io.BufferedWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class KfCV {
	public static void kfcv (Instances data, NaiveBayes nb, BufferedWriter writer) throws Exception {
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(nb, data, 5, new Random(42));
		
		writer.write(eval.toMatrixString("\n\n===== CROSS VALIDATION NAHASMEN MATRIZEA ====="));
	}
}
