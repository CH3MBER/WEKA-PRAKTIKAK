package stratHoldOut;

import java.io.File;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

public class Main {
	public static void main(String[] args) throws Exception {
		// DATUAK LORTU
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1) {data.setClassIndex(data.numAttributes() - 1);}
		
		// STRATIFIED HOLD-OUT EGIN
		data.randomize(new Random(42));
		StratifiedRemoveFolds srf = new StratifiedRemoveFolds();
		srf.setNumFolds(5);
		srf.setFold(1);
		srf.setInputFormat(data);
		Instances test = Filter.useFilter(data, srf);
		srf.setInvertSelection(true);
		srf.setInputFormat(data);
		Instances train = Filter.useFilter(data, srf);
		
		// TRAIN ETA TEST FITXATEGIAK GORDE
		ArffSaver saver = new ArffSaver();
		saver.setInstances(train);
		saver.setFile(new File(args[1]));
		saver.writeBatch();
		
		saver.setInstances(test);
		saver.setFile(new File(args[2]));
		saver.writeBatch();
		
		TrainVSDev.trainVSDev(test, train, data);
		System.out.println("\n\n\n\n (!!!) FITXATEGIAK SORTUTA (!!!) \n\n\n\n");
	}
}
