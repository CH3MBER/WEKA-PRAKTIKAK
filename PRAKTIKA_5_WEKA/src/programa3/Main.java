package programa3;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Main {
	public static void main (String[] args) throws Exception {
		DataSource source = new DataSource(args[0]);
		Instances test = source.getDataSet();
		test.setClassIndex(test.numAttributes()-1);
		
		Object[] stored = (Object[]) weka.core.SerializationHelper.read(args[1]);
		//Classifier cls = (Classifier) stored[0];
		Instances trainHeader = (Instances) stored[1];
		
		// GOIBURUAK DESBERDINAK BADIRA
		/*if (!test.equalHeaders(trainHeader)) {
			for (int i = 0; i < test.numAttributes(); i++) {
				if (!test.attribute(i).name().equals(trainHeader.attribute(i).name())) {
					Remove rm = new Remove();
					rm.setAttributeIndices(Integer.toString(i));
					rm.setInputFormat(test);
					test = Filter.useFilter(test, rm);
				}
			}
		}*/
		
		if (!test.equalHeaders(trainHeader)) {
		    StringBuilder toRemove = new StringBuilder();

		    for (int i = 0; i < test.numAttributes(); i++) {
		        String attName = test.attribute(i).name();
		        boolean inTrainHeader = (trainHeader.attribute(attName) != null);
		        boolean isClass       = (i == test.classIndex());

		        if (!inTrainHeader && !isClass) {
		            if (toRemove.length() > 0) {toRemove.append(",");}
		            toRemove.append(i + 1); 
		        }
		    }

		    if (toRemove.length() > 0) {
		        Remove rm = new Remove();
		        rm.setAttributeIndices(toRemove.toString());
		        rm.setInputFormat(test);
		        test = Filter.useFilter(test, rm);
		        test.setClassIndex(test.numAttributes() - 1);
		    }
		}
		
	}
}
