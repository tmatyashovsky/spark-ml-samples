package com.lohika.morning.ml.spark.driver.service;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.junit.Test;

public class Word2VecTest extends BaseTest {

    @Test
    public void should() {
        Dataset<String> files = getSparkSession().read().textFile("/Users/tmatyashovsky/Workspace/text/TEXTS/Spice_Girls").repartition(1);
        Dataset<Row> replaced = files.withColumn("value", functions.regexp_replace(files.col("value"), "&quot;", "'"));
        replaced = replaced.withColumn("value", functions.regexp_replace(replaced.col("value"), "�", "'"));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("[")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("CHORUS")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("chorus")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("Chorus")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("VERSE")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("verse")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("Verse")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("(Chorus)")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("(chorus)")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("-Chorus-")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("(rpt 1)")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("(Bridge)")));
        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("(bridge)")));
//        replaced = replaced.filter(functions.not(replaced.col("value").startsWith("(vocalizes)")));

        replaced.write().text("/Users/tmatyashovsky/Downloads/sg");

        System.out.println("test");

//        getSparkSession().createDataFrame(files.values().rdd(), new StructType(new StructField[]{new StructField("value", DataTypes.StringType, true, Metadata.empty())}));
    }

}
