package programa2;


import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class Main {
	public static void main (String[] args) throws Exception {
		DataSource source = new DataSource(args[0]);
		Instances train = source.getDataSet();
		train.setClassIndex(train.numAttributes()-1);

		// ATRIBUTE SELECTION ERABILI
		AttributeSelection as = new AttributeSelection();
		as.setInputFormat(train);
		train = Filter.useFilter(train, as);
		
		// NAIVE BAYES EREDUA GORDE 
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(train);
		
		Instances trainHeader = new Instances(train, 0);
		weka.core.SerializationHelper.write(args[1], new Object[]{nb, trainHeader});
		
		System.out.println("\n\t (!!!) FITXATEGIA SORTUTA (!!!) \n");
		
	}
}

