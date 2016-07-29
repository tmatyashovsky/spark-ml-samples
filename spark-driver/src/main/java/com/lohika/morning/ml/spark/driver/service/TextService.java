package com.lohika.morning.ml.spark.driver.service;

import com.lohika.morning.ml.spark.distributed.library.function.map.generic.ConvertMLLabeledPointToMLlibLabeledPoint;
import com.lohika.morning.ml.spark.distributed.library.function.map.generic.ConvertMLVectorToMLlibVector;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import scala.Tuple2;

@Component
public class TextService {

    @Autowired
    private SparkSession sparkSession;

    @Autowired
    private MLlibUtilityService mllibutilityservice;

    @Autowired
    private MLlibService mLlibService;

    public void classifyDarkLyrics(final String inputDirectory) {
        JavaRDD<LabeledPoint> trainingSet = getTrainingSet(inputDirectory);

        Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> datasets = mLlibService.getTrainingAndTestDatasets(trainingSet);

        LogisticRegressionModel logisticRegressionModel = mLlibService.trainLogisticRegression(datasets._1(), datasets._2(), 2);

        // Input data: Each row is a bag of words from a sentence or document.
        List<Row> unknownLyrics = Arrays.asList(
                RowFactory.create(Arrays.asList("Dirty women".split(" "))),
                RowFactory.create(Arrays.asList("Justice all".split(" "))),
                RowFactory.create(Arrays.asList("Heaven and hell".split(" ")))
        );
        StructType schema = new StructType(new StructField[]{
            new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        Dataset<Row> unknownLyricsDataset = sparkSession.createDataFrame(unknownLyrics, schema);

        // Learn a mapping from words to Vectors.
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("text")
                .setOutputCol("features")
                .setVectorSize(20)
                .setMinCount(0);
        Word2VecModel word2VecModel = word2Vec.fit(unknownLyricsDataset);
        Dataset<Row> vector = word2VecModel.transform(unknownLyricsDataset);

        System.out.print(logisticRegressionModel.predict(mlVectorToMLlibVector(vector)).collect());
    }

    private JavaRDD<LabeledPoint> getTrainingSet(String inputDirectory) {
        Dataset<String> blackSabbathSencences = sparkSession.read().text(Paths.get(inputDirectory).resolve("black_sabbath.txt").toString());
        Dataset<String> metallicaSentences = sparkSession.read().text(Paths.get(inputDirectory).resolve("metallica.txt").toString());
        blackSabbathSencences.withColumn("label", functions.lit(0D));
        metallicaSentences.withColumn("label", functions.lit(0D));

        Dataset<String> sentences = blackSabbathSencences.union(metallicaSentences);
        Dataset<Row> sentencesWithIds = sentences.withColumn("id", functions.monotonically_increasing_id());

        Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words");
        Dataset<Row> words = tokenizer.transform(sentencesWithIds);

        Dataset<Row> separateWords = words.withColumn("words", functions.explode(words.col("words")));
        separateWords = separateWords.withColumn("words", functions.array(separateWords.col("words")));

        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("words")
                .setOutputCol("features")
                .setVectorSize(20)
                .setMinCount(0);

        Word2VecModel word2VecModel = word2Vec.fit(separateWords);
        Dataset<Row> featuresPerWord = word2VecModel.transform(separateWords);

        Dataset<Row> featuresPerSentence = featuresPerWord.groupBy(featuresPerWord.col("id")).mean("features.values");

        return mlLabeledPointToMLlibLabeledPoint(featuresPerSentence);
    }

    public JavaRDD<LabeledPoint> mlLabeledPointToMLlibLabeledPoint(Dataset<Row> mlLabeledPointDataset) {
        return mlLabeledPointDataset.javaRDD().map(new ConvertMLLabeledPointToMLlibLabeledPoint());
    }

    public JavaRDD<Vector> mlVectorToMLlibVector(Dataset<Row> mlVector) {
        return mlVector.javaRDD().map(new ConvertMLVectorToMLlibVector());
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
