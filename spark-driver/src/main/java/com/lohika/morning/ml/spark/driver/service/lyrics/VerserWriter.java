package com.lohika.morning.ml.spark.driver.service.lyrics;

import org.apache.spark.ml.util.MLWriter;

public class VerserWriter extends MLWriter {

    private Verser verser;

    public VerserWriter(Verser verser) {
        this.verser = verser;
    }

    @Override
    public void saveImpl(String path) {
        // Save metadata and params.
        // Save model data, in our case one additional custom parameter.
//        List<Row> modelData = Collections.singletonList(
//                RowFactory.create(verser.sentencesInVerse().parent(),
//                        verser.sentencesInVerse().name(),
//                        verser.sentencesInVerse().doc(),
//                        verser.getSentencesInVerse()));
//        StructType schema = new StructType(new StructField[]{
//            new StructField("parent", DataTypes.StringType, true, Metadata.empty()),
//            new StructField("name", DataTypes.StringType, true, Metadata.empty()),
//            new StructField("doc", DataTypes.StringType, true, Metadata.empty()),
//            new StructField("value", DataTypes.IntegerType, true, Metadata.empty())
//        });
//
//        Path dataPath = Paths.get(path).resolve("data");
//        sqlContext().createDataFrame(modelData, schema).repartition(1).write().parquet(dataPath.toString());
    }
}
