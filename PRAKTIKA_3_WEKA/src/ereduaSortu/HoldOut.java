package ereduaSortu;

import java.io.BufferedWriter;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class HoldOut {
	public static void holdOut(Instances data, NaiveBayes nb, BufferedWriter writer) throws Exception{
		RemovePercentage rp = new RemovePercentage();
		rp.setPercentage(70);
		rp.setInputFormat(data);
		Instances test = Filter.useFilter(data, rp);
		
		Evaluation eval = new Evaluation(data.stringFreeStructure());
		eval.evaluateModel(nb, test);
		
		writer.write(eval.toMatrixString("\n\n===== HOLD-OUT NAHASMEN MATRIZEA ====="));
	}
}
