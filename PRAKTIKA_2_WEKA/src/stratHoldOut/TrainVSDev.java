package stratHoldOut;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.time.LocalDate;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class TrainVSDev {
	public static void trainVSDev (Instances test, Instances train, Instances data) throws Exception {
		// ATAL HONETAN KONKRETUKI ESKATZEN DENA BETETZEKO, BI DATASOURCE SARTU ETA 
		// TRAIN ETA TEST ATRIBUTUAK SORTU, ONDOREN EBALUAZIOA EGITEKO.
		NaiveBayes nb = new NaiveBayes();
		Evaluation eval = new Evaluation(data);
		nb.buildClassifier(train);
		eval.evaluateModel(nb, test);
		
		// ATAL HONETAN KONKRETUKI ESKATZEN DENA BETETZEKO, "FileWriter(args[2])" IDATZI.
		BufferedWriter writer = new BufferedWriter
									(new FileWriter("/home/gdivasson/Documentos/evaluation.txt"));
		
		writer.write("EXEKUZIO DATA: " + LocalDate.now());
		writer.write("\nARGUMENTUAK: [ARGUMENTUAK]");
		writer.write(eval.toMatrixString("\n\n===== NAHASMEN MATRIZEA ====="));
		writer.write("\nACCURACY: %" + eval.pctCorrect());
		writer.flush();
		writer.close();
	}
}
