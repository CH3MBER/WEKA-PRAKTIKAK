package praktika;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.BallTree;
import weka.core.neighboursearch.CoverTree;
import weka.core.neighboursearch.FilteredNeighbourSearch;
import weka.core.neighboursearch.KDTree;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class Main {
	public static void main (String[] args) throws Exception {
		// DATUA HARTU
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if (data.classIndex()==-1) {data.setClassIndex(data.numAttributes()-1);}
		
		// PARAMETROAK SORTU
		NearestNeighbourSearch[] d = {new BallTree(), new CoverTree(), new FilteredNeighbourSearch(),
									  new KDTree(), new LinearNNSearch()};
		SelectedTag[] w = {new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING), 
						   new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING),
						   new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING)};
		int kOnena = 0;
		NearestNeighbourSearch dOnena = null;
		SelectedTag wOnena = null;
		double baloreOnena = 0;
		
		// HOLD OUT BANAKETA
		Resample rsp = new Resample();
		rsp.setRandomSeed(42);
		rsp.setNoReplacement(true);
		rsp.setSampleSizePercent(66);
		rsp.setInvertSelection(false);
		rsp.setInputFormat(data);
		Instances train = Filter.useFilter(data, rsp);
		rsp.setInvertSelection(true);
		rsp.setInputFormat(data);
		Instances test = Filter.useFilter(data, rsp);
		
		for (int ki = 1; ki < 10; ki++) {
			for (int di = 0; di < d.length; di++) {
				for (int wi = 0; wi < w.length; wi++) {
					IBk ibk = new IBk();
					ibk.setKNN(ki);
					ibk.setNearestNeighbourSearchAlgorithm(d[di]);
					ibk.setDistanceWeighting(w[wi]);
					ibk.buildClassifier(train);
					
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(ibk, test);
					
					if (eval.weightedFMeasure() > baloreOnena) {
						baloreOnena = eval.weightedFMeasure();
						kOnena = ki;
						dOnena = d[di];
						wOnena = w[wi];
					}
				}
			}
		}
		
		System.out.println("----------------------- HOLD OUT -----------------------");
		System.out.println("\n=== PARAMETRO ONENAK ===");
		System.out.println("K: " + kOnena);
		System.out.println("D: " + dOnena.getClass().getSimpleName());
		System.out.println("W: " + wOnena.getSelectedTag().getReadable());
		
		baloreOnena = 0;
		for (int ki = 1; ki < 10; ki++) {
			for (int di = 0; di < d.length; di++) {
				for (int wi = 0; wi < w.length; wi++) {
					IBk ibk = new IBk();
					ibk.setKNN(ki);
					ibk.setNearestNeighbourSearchAlgorithm(d[di]);
					ibk.setDistanceWeighting(w[wi]);
					
					Evaluation eval = new Evaluation(data);
					eval.crossValidateModel(ibk, data, 3, new Random(42));
					
					if (eval.pctCorrect() > baloreOnena) {
						baloreOnena = eval.pctCorrect();
						kOnena = ki;
						dOnena = d[di];
						wOnena = w[wi];
					}
				}
			}
		}
		
		System.out.println("\n---------------- K FOLD CROSS VALIDATION ----------------");
		System.out.println("\n=== PARAMETRO ONENAK ===");
		System.out.println("K: " + kOnena);
		System.out.println("D: " + dOnena.getClass().getSimpleName());
		System.out.println("W: " + wOnena.getSelectedTag().getReadable());

		/*CVParameterSelection cvps = new CVParameterSelection();
		cvps.setClassifier(ibk);
		cvps.addCVParameter("K 1 10 10");
		cvps.buildClassifier(data);
		System.out.println(cvps.toString());*/	
	}
}
