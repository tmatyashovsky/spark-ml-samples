package com.lohika.morning.ml.spark.driver.service;

import com.lohika.morning.ml.spark.distributed.library.function.map.generic.DenseVectorValuesElementsAverageAggregator;
import com.lohika.morning.ml.spark.distributed.library.function.map.generic.DoubleArrayAVGHolder;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.mllib.feature.Stemmer;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Aggregator;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class Word2VecTest extends BaseTest {

    @Test
    public void word2VecTest() {
        Dataset<Row> unknownLyricsDataset = getUnknownLyrics();

        unknownLyricsDataset = unknownLyricsDataset.withColumn("id", functions.monotonically_increasing_id());
        Dataset<Row> sentences = unknownLyricsDataset.withColumn("sentences", functions.split(unknownLyricsDataset.col("value"), " "));

        Word2Vec word2VecUsingSentences = new Word2Vec()
                .setInputCol("sentences")
                .setOutputCol("features")
                .setVectorSize(100)
                .setMinCount(0);

        Word2VecModel word2VecModel1 = word2VecUsingSentences.fit(sentences);
        Dataset<Row> features1 = word2VecModel1.transform(sentences);

        List<Row> vectors1 = word2VecModel1.getVectors().select("word").sort(word2VecModel1.getVectors().col("word")).collectAsList();

        System.out.println("Features: " + features1.first().getAs("features"));

        Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words");
        Dataset<Row> words = tokenizer.transform(unknownLyricsDataset);

        Dataset<Row> separatedWords = words.withColumn("words", functions.explode(words.col("words")));
        separatedWords = separatedWords.withColumn("words", functions.array(separatedWords.col("words")));

        Word2Vec word2VecUsingWords = new Word2Vec()
                .setInputCol("words")
                .setOutputCol("features")
                .setVectorSize(100)
                .setMinCount(0);

        // Fit using words.
        Word2VecModel word2VecModel2 = word2VecUsingWords.fit(separatedWords);

        List<Row> vectors2 = word2VecModel2.getVectors().select("word").sort(word2VecModel2.getVectors().col("word")).collectAsList();

        assertEquals(vectors1, vectors2);

        // But transform using sentences.
        word2VecModel2.setInputCol("sentences");
        Dataset<Row> features2 = word2VecModel2.transform(sentences);

        System.out.println("Features: " + features2.first().getAs("features"));

        // Transform using words.
        word2VecModel2.setInputCol("words");
        Dataset<Row> features3 = word2VecModel2.transform(separatedWords);

        Aggregator<Row, DoubleArrayAVGHolder, Row> averageAggregator = new DenseVectorValuesElementsAverageAggregator(100, false);
        Dataset<Row> featuresPerSentence = features3.groupBy("id").agg(averageAggregator.toColumn().as("sentenceFeatures"))
                .select("id", "sentenceFeatures.label", "sentenceFeatures.averageValues");

        System.out.println("Features: " + featuresPerSentence.first().getAs("averageValues"));
    }

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

}
