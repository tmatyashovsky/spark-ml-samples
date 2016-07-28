package com.lohika.morning.ml.spark.driver.service;

import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class TextService {

    @Autowired
    private SparkSession sparkSession;

    public void classifyDarkLyrics(final String textFile) {
        Dataset<String> sentences = sparkSession.read().text(textFile);

        Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words");
        Dataset<Row> words = tokenizer.transform(sentences);

        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("words")
                .setOutputCol("features")
                .setVectorSize(10)
                .setMinCount(0);

        Word2VecModel word2VecModel = word2Vec.fit(words);
        Dataset<Row> features = word2VecModel.transform(words);


    }

    // BACKUP
//    Dataset<String> text = sparkSession.read().text(textFile);
//    Dataset<Row> sentences = text.withColumn("id", functions.monotonically_increasing_id());
//
//    Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words");
//    Dataset<Row> words = tokenizer.transform(sentences);
//
//    StopWordsRemover stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered");
//    Dataset<Row> filtered = stopWordsRemover.transform(words);
//
//    Column filteredArray = functions.explode(filtered.col("filtered"));
//    Dataset<Row> filteredWords = filtered.withColumn("filtered", filteredArray);
//
//    Dataset<Row> stemmed = new Stemmer()
//            .setInputCol("filtered")
//            .setOutputCol("stemmed")
//            .setLanguage("English")
//            .transform(filteredWords);
//
//    Column stemmedArray = functions.array(stemmed.col("stemmed"));
//    Dataset<Row> stemmedWords = stemmed.withColumn("stemmedWords", stemmedArray);
//
//    Word2VecModel word2VecModel = new Word2Vec().setInputCol("stemmedWords").setOutputCol("features").setVectorSize(3).fit(stemmedWords);
//    Dataset<Row> features = word2VecModel.transform(stemmedWords);
//
//    System.out.println();

}
