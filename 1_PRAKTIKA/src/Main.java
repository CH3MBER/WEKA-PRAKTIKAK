// ====================NO AI USED, MADE BY HUMAN====================
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.AttributeStats;
import java.util.Enumeration;

public class Main {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("/home/gdivasson/Documentos/weka/data/heart-c.arff");
        Instances data = source.getDataSet();
        AttributeStats atributua = new AttributeStats();
        atributua = data.attributeStats(0);
        if (data.classIndex() == -1) {data.setClassIndex(data.numAttributes() - 1);}
        System.out.println();
        System.out.println("========== PATH ==========");
        System.out.println("/home/gdivasson/Documentos/weka/data/heart-c.arff");
        System.out.println();
        System.out.println("INSTATZIA KOPURUA: " + data.numInstances());
        System.out.println("ATRIBUTU KOPURUA: " + data.numAttributes());
        System.out.println("LEHENENGO ATRIBUTUAK HAR DITZAKEEN BALIO EZBERDINAK: " + atributua.distinctCount);
        System.out.println("AZKEN ATRIBUTUAK HARTZEN DITUEN BALIOAK: ");
        atributua = data.attributeStats(13);
        int[] balioak = atributua.nominalCounts;
        int i = 0;
        Enumeration<Object> azkenDatuak = data.attribute(13).enumerateValues();
        while (azkenDatuak.hasMoreElements()) {
            System.out.println(azkenDatuak.nextElement() + ": " + balioak[i]);
            i++;
        }
        atributua = data.attributeStats(12);
        System.out.println("AZKEN AURREKO ATRIBUTUAK MISSING VALUES: " + atributua.missingCount);
        System.out.println();
        System.out.println("========== ZENBAKIZKOAK (AGE) ==========");
        atributua = data.attributeStats(0);
        System.out.println(atributua.numericStats);
    }
}