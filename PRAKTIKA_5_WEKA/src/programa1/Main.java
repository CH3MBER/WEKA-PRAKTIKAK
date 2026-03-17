package programa1;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class Main {
	public static void main (String[] args) throws Exception {
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes()-1);
		
		// RESAMPLE BIDEZ ZATITU 70/30
		Resample rsp = new Resample();
		rsp.setRandomSeed(42);
		rsp.setSampleSizePercent(70);
		rsp.setNoReplacement(true); 
		rsp.setInputFormat(data);
		Instances train = Filter.useFilter(data, rsp);
		rsp.setInvertSelection(true);
		rsp.setInputFormat(data);
		Instances test = Filter.useFilter(data, rsp);
		
		// TEST ITSUA BIHURTU
		for (int i = 0; i < test.numInstances(); i++) {
			test.instance(i).setClassMissing();
		}	
		
		// TRAIN ETA TEST GORDE
		ArffSaver saver = new ArffSaver();
		saver.setInstances(train);
		saver.setFile(new File(args[1]));
		saver.writeBatch();
		saver = new ArffSaver();
		saver.setInstances(test);
		saver.setFile(new File(args[2]));
		saver.writeBatch();
		
		
		System.out.println("\n\t(!!!) FITXATEGIA SORTUTA (!!!)\n");
	}
}
