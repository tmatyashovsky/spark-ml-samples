package com.lohika.morning.ml.spark.driver.service;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.feature.Stemmer;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

public class Word2VecTest extends BaseTest {

    private Dataset<Row> getUnknownLyrics() {
        List<Row> unknownLyrics = Arrays.asList(
                RowFactory.create("The \"day\" that never, comes.!"),
                RowFactory.create("Is this the end of the beginning"),
                RowFactory.create("Lost in time I wonder will my ship be found"),
                RowFactory.create("Finished with my woman cause she could not help me with my mind"),
                RowFactory.create("So I dub the unforgiven"),
                RowFactory.create("Give me baby one more time")
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("value", DataTypes.StringType, true, Metadata.empty())
        });

        return getSparkSession().createDataFrame(unknownLyrics, schema);
    }

    @Test
    public void secondTest() {
        Dataset<Row> unknownLyricsDataset = getUnknownLyrics();

        unknownLyricsDataset = unknownLyricsDataset.withColumn("label", functions.lit(0D));
        unknownLyricsDataset = unknownLyricsDataset.withColumn("id", functions.monotonically_increasing_id());
        unknownLyricsDataset = unknownLyricsDataset.withColumn("rowNumber", functions.row_number().over(Window.orderBy("id")));
        unknownLyricsDataset = unknownLyricsDataset.withColumn("value", functions.regexp_replace(unknownLyricsDataset.col("value"), "[^\\w\\s]", ""));

        Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words");
        Dataset<Row> words = tokenizer.transform(unknownLyricsDataset);

        StopWordsRemover stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered");
        Dataset<Row> filtered = stopWordsRemover.transform(words);

        Column filteredArray = functions.explode(filtered.col("filtered"));
        Dataset<Row> filteredWords = filtered.withColumn("filtered", filteredArray);

        Dataset<Row> stemmedWords = new Stemmer()
                .setInputCol("filtered")
                .setOutputCol("stemmed")
                .setLanguage("English")
                .transform(filteredWords);

        stemmedWords = stemmedWords.groupBy("rowNumber", "value", "label").agg(functions.column("rowNumber"), functions.concat_ws(" ", functions.collect_list("stemmed")).as("stemmedConcat"));

        Column newSessionExpression = functions
                .when(
                        functions
                                .column("rowNumber")
                                .mod(3)
                                .equalTo(1),
                        1
                )
                .otherwise(0);
        Dataset<Row> testSequence = stemmedWords.withColumn("seqStart", newSessionExpression);

//        testSequence.orderBy("id").show();

        Dataset<Row> finalDF = testSequence
                .withColumn("groupID",
                        functions
                                .sum("seqStart")
                                .over(
                                        Window
                                                .orderBy("rowNumber")
                                                .rowsBetween(Long.MIN_VALUE, 0)
                                )
                )
                .select("rowNumber", "stemmedConcat", "groupID", "label");

        finalDF = finalDF.groupBy("groupID").agg(
                functions.first("label"),
                functions.split(functions.concat_ws(" ", functions.collect_list(functions.column("stemmedConcat"))), " ").as("verses")
        );

//        .map(new VerseMapFunction(),
//                RowEncoder.apply(
////                        new StructType(new StructField[]{DataTypes.createStructField("verse", DataTypes.createArrayType(DataTypes.StringType), false)}))

        finalDF.collectAsList();

        System.out.println("test");
    }

    @Test
    public void should() {
        Dataset<String> files = getSparkSession().read().textFile("/Users/tmatyashovsky/Workspace/text/TEXTS/Sepultura").repartition(1);
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

        replaced.write().text("/Users/tmatyashovsky/Downloads/sepultura");

        System.out.println("test");

//        getSparkSession().createDataFrame(files.values().rdd(), new StructType(new StructField[]{new StructField("value", DataTypes.StringType, true, Metadata.empty())}));
    }

}
