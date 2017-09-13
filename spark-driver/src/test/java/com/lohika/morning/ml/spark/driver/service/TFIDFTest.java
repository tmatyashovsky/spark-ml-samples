package com.lohika.morning.ml.spark.driver.service;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

public class TFIDFTest extends BaseTest {

    @Test
    public void test() {
        List<Row> data = Arrays.asList(
                RowFactory.create(0.0, "Hi I heard about Spark"),
                RowFactory.create(0.0, "I wish Java could use case classes"),
                RowFactory.create(1.0, "Logistic regression models are neat")
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
        });
        Dataset<Row> sentenceData = getSparkSession().createDataFrame(data, schema);

        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
        Dataset<Row> wordsData = tokenizer.transform(sentenceData);

        System.out.println("Words:\n");
        System.out.println(wordsData.select("label", "words").collectAsList());
        System.out.println("----------------");

        int numFeatures = 5;
        HashingTF hashingTF = new HashingTF()
                .setInputCol("sentence")
                .setOutputCol("rawFeatures")
                .setNumFeatures(numFeatures);

        Dataset<Row> featurizedData = hashingTF.transform(wordsData);
        System.out.println("Raw Features:\n");
        System.out.println(featurizedData.select("label", "rawFeatures").collectAsList());
        System.out.println("----------------");

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(featurizedData);

        Dataset<Row> rescaledData = idfModel.transform(featurizedData);

        System.out.println("Features:\n");
        System.out.println(rescaledData.select("label", "features").collectAsList());
        System.out.println("----------------");
    }

}
