package p1;

import java.util.Enumeration;

import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
	public static void main(String[] args) throws Exception {
		// DATUAK LORTU
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		
		// DATUAREN INFORMAZIOA PANTAILARATU
		System.out.println("1. FITXATEGIKO PATH-A: " + args[0]);
		System.out.println("2. INSTANTZIA KOPURUA: " + data.numInstances());
		System.out.println("3. ATRIBUTU KOPURUA: " + data.numAttributes());
		System.out.println("4. LEHENENGO ATRIBUTUKO BALIO EZBERDINAK: " + 
							data.attributeStats(0).distinctCount);
		
		// AZKEN ATRIBUTUKO IDENTIFIKATZAILE ETA BALIOAK PANTAILARATU
		System.out.println("5. AZKEN ATRIBUTUKO BALIOAK: ");
		AttributeStats klasea = data.attributeStats(data.classIndex());
		Enumeration<Object> klaseIdent = data.attribute(data.classIndex()).enumerateValues();
		for (int i = 0; i < klasea.nominalCounts.length; i++) {
			System.out.println("   -" + klaseIdent.nextElement() + ": " + klasea.nominalCounts[i]);
		}
		
		// KLASE MINORITARIOA LORTU
		int minBalore = Integer.MAX_VALUE;
		int minIndex = 0;
		for (int i = 0; i < klasea.nominalCounts.length; i++) {
			if (klasea.nominalCounts[i] < minBalore) {
				minIndex = i;
				minBalore = klasea.nominalCounts[i];
			}
		}
		System.out.println("6. KLASE MINORITARIOA: " + 
							data.attribute(data.classIndex()).value(minIndex));
		System.out.println("7. AZKEN AURREKO ATRIBUTUKO MISSING VALUES: " + 
							data.attributeStats(data.classIndex() - 1).missingCount);
	}
}
