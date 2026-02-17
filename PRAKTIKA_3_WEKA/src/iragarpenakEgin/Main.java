package iragarpenakEgin;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
	public static void main(String[] args) throws Exception {
		Classifier nb = (Classifier) weka.core.SerializationHelper.read(args[0]);
		
		DataSource source = new DataSource(args[1]);
		Instances testBlind = source.getDataSet();
		if(testBlind.classIndex() == -1) {testBlind.setClassIndex(testBlind.numAttributes() - 1);}
		
		StringBuffer iragarpenak = new StringBuffer();
        PlainText output = new PlainText();
        output.setBuffer(iragarpenak);
        output.setHeader(testBlind); 
        output.printHeader();
        
        Evaluation eval = new Evaluation(testBlind.stringFreeStructure());
        eval.evaluateModel(nb, testBlind, output);
        
        BufferedWriter writer = new BufferedWriter(new FileWriter(args[2]));
        writer.write(iragarpenak.toString());	
		writer.flush();	
		writer.close();
		
		System.out.println("\n\n\n\n (!!!) FITXATEGIA SORTUTA (!!!) \n\n\n\n");
	}
}
