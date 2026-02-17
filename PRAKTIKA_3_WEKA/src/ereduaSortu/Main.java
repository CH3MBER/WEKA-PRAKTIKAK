package ereduaSortu;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.time.LocalDate;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
	public static void main(String[] args) throws Exception {
		// DATUAK LORTU
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1) {data.setClassIndex(data.numAttributes()-1);}
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(data);
		
		weka.core.SerializationHelper.write(args[1], nb);
		
		BufferedWriter writer = new BufferedWriter(
									new FileWriter(args[2]));
		
		// KFCV ETA HOLDOUT EGIN
		writer.write("EXEKUZIO DATA: " + LocalDate.now());
		writer.write("\nARGUMENTUAK: " + args[0] + " " + args[1] + " " + args[2]);
		KfCV.kfcv(data, nb, writer);
		HoldOut.holdOut(data, nb, writer);
		writer.flush();
		writer.close();
		
		System.out.println("\n\n\n\n (!!!) FITXATEGIA SORTUTA (!!!) \n\n\n\n");
	}
}
