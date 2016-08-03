package com.lohika.morning.ml.spark.driver.service;

import com.lohika.morning.ml.spark.distributed.library.function.map.generic.DenseVectorValuesElementsAverageAggregator;
import com.lohika.morning.ml.spark.distributed.library.function.map.generic.DoubleArrayAVGHolder;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.expressions.Aggregator;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

public class Word2VecTest extends BaseTest {

    @Test
    public void word2VecTest() {
        List<Row> unknownLyrics = Arrays.asList(
                RowFactory.create("The day that never comes"),
                RowFactory.create("Is this the end of the beginning"),
                RowFactory.create("Lost in time I wonder will my ship be found")
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("value", DataTypes.StringType, true, Metadata.empty())
        });

        Dataset<Row> unknownLyricsDataset = getSparkSession().createDataFrame(unknownLyrics, schema);
        unknownLyricsDataset = unknownLyricsDataset.withColumn("id", functions.monotonically_increasing_id());

        Dataset<Row> sentences = unknownLyricsDataset.withColumn("sentences", functions.array(unknownLyricsDataset.col("value")));

        Word2Vec word2VecUsingSentences = new Word2Vec()
                .setInputCol("sentences")
                .setOutputCol("features")
                .setVectorSize(100)
                .setMinCount(0);

        Word2VecModel word2VecModel = word2VecUsingSentences.fit(sentences);
        Dataset<Row> features1 = word2VecModel.transform(sentences);

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
        word2VecModel = word2VecUsingWords.fit(separatedWords);

        // But transform using sentences.
        word2VecModel.setInputCol("sentences");
        Dataset<Row> features2 = word2VecModel.transform(sentences);

        System.out.println("Features: " + features2.first().getAs("features"));

        // Transform using words.
        word2VecModel.setInputCol("words");
        Dataset<Row> features3 = word2VecModel.transform(separatedWords);

        Aggregator<Row, DoubleArrayAVGHolder, Row> averageAggregator = new DenseVectorValuesElementsAverageAggregator(100, false);
        Dataset<Row> featuresPerSentence = features3.groupBy("id").agg(averageAggregator.toColumn().as("sentenceFeatures"))
                .select("id", "sentenceFeatures.label", "sentenceFeatures.averageValues");

        System.out.println("Features: " + featuresPerSentence.first().getAs("averageValues"));
    }

}
