package com.lohika.morning.ml.spark.driver.service;

import com.lohika.morning.ml.spark.distributed.library.function.map.generic.*;
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
import org.apache.spark.sql.expressions.Aggregator;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import scala.Tuple2;

@Component
public class TextService {

    private int vectorSize = 100;

    @Autowired
    private SparkSession sparkSession;

    @Autowired
    private MLlibService mLlibService;

    public void classifyDarkLyrics(final String inputDirectory) {
        Tuple2<JavaRDD<LabeledPoint>, Word2VecModel> trainingSetAndWord2Vec = getTrainingSet(inputDirectory);

        Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> datasets = mLlibService.getTrainingAndTestDatasets(trainingSetAndWord2Vec._1());

        LogisticRegressionModel logisticRegressionModel = mLlibService.trainLogisticRegression(datasets._1(), datasets._2(), 2);

//        System.out.print(logisticRegressionModel.predict(mlVectorToMLlibVector(getValidationSet())).collect());
    }

    private Tuple2<JavaRDD<LabeledPoint>, Word2VecModel> getTrainingSet(String inputDirectory) {
        Dataset<String> blackSabbathLyrics = getLyrics(inputDirectory, "black_sabbath.txt");
        System.out.println("Black sabbath sentences = " + blackSabbathLyrics.count());

        Dataset<String> metallicaLyrics = getLyrics(inputDirectory, "metallica.txt");
        System.out.println("Metallica sentences = " + metallicaLyrics.count());

//        Dataset<String> beatlesLyrics = getLyrics(inputDirectory, "beatles.txt");
//        System.out.println("Beatles sentences = " + beatlesLyrics.count());

        Dataset<Row> blackSabbathSentences = blackSabbathLyrics.withColumn("label", functions.lit(0D));
        Dataset<Row> metallicaSentences = metallicaLyrics.withColumn("label", functions.lit(1D));
//        Dataset<Row> beatlesSentences = beatlesLyrics.withColumn("label", functions.lit(2D));

        Dataset<Row> sentences = blackSabbathSentences.union(metallicaSentences);

        Dataset<Row> separatedWords = getWord2VecDataset(sentences);

        System.out.println("Words for Word2Vec = " + separatedWords.select("words").distinct().count());

        // Create model.
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("words")
                .setOutputCol("features")
                .setVectorSize(vectorSize)
                .setMinCount(0);

        // Fit model.
        Word2VecModel word2VecModel = word2Vec.fit(separatedWords);

        // Train words and get full features.
        Dataset<Row> fullFeatures = getFullFeatures(separatedWords, word2VecModel);

        return new Tuple2<>(mlLabeledPointToMLlibLabeledPoint(fullFeatures), word2VecModel);
    }

    private Dataset<Row> getWord2VecDataset(Dataset<Row> sentences) {
        // Add unique id to each sentence of lyrics.
        Dataset<Row> sentencesWithIds = sentences.withColumn("id", functions.monotonically_increasing_id());

        // Split into words.
        Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words");
        Dataset<Row> words = tokenizer.transform(sentencesWithIds);

        // Create as many row as words. Using this for Word2Vec.
        Dataset<Row> separatedWords = words.withColumn("words", functions.explode(words.col("words")));
        // Wrap string into array. This is a requirement for Word2Vec.
        separatedWords = separatedWords.withColumn("words", functions.array(separatedWords.col("words")));

        return separatedWords;
    }

    private Dataset<Row> getValidationSet(Word2VecModel word2VecModel) {
        List<Row> unknownLyrics = Arrays.asList(
                RowFactory.create("The day that never comes"),
                RowFactory.create("Is this the end of the beginning"),
                RowFactory.create("Lost in time I wonder will my ship be found")
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("value", DataTypes.StringType, true, Metadata.empty())
        });

        Dataset<Row> unknownLyricsDataset = sparkSession.createDataFrame(unknownLyrics, schema);
        Dataset<Row> separatedWords = getWord2VecDataset(unknownLyricsDataset);

        return getFullFeatures(separatedWords, word2VecModel);
    }

    private Dataset<String> getLyrics(String inputDirectory, String fileName) {
        Dataset<String> lyrics = sparkSession.read().textFile(Paths.get(inputDirectory).resolve(fileName).toString());
        lyrics = lyrics.filter(lyrics.col("value").notEqual(""));
        lyrics = lyrics.filter(lyrics.col("value").contains(" "));

        return lyrics;
    }

    private Dataset<Row> getFullFeatures(Dataset<Row> separatedWords, Word2VecModel word2VecModel) {
        Dataset<Row> wordsAsFeatures = word2VecModel.transform(separatedWords);

        Dataset<Row> averagePerSentence = getAverages(wordsAsFeatures);

        Dataset<Row> joined = wordsAsFeatures.join(averagePerSentence, "id");

        return getVariances(joined);
    }

    public Dataset<Row> getAverages(Dataset<Row> dataset) {
        Aggregator<Row, DoubleArrayAVGHolder, Row> averageAggregator = new DenseVectorValuesElementsAverageAggregator(vectorSize, true);

        return dataset.groupBy("id").agg(averageAggregator.toColumn().as("averagePerSentence"))
                .select("id", "averagePerSentence.label", "averagePerSentence.averages");
    }

    public Dataset<Row> getVariances(Dataset<Row> dataset) {
        Aggregator<Row, DoubleArrayVarianceHolder, Row> varianceAggregator = new DenseVectorValuesElementsVarianceAggregator(vectorSize);

        return dataset.groupBy("id").agg(varianceAggregator.toColumn().as("fullFeatures"))
                .select("id", "fullFeatures.label", "fullFeatures.averages", "fullFeatures.variances");
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


    public void setVectorSize(int vectorSize) {
        this.vectorSize = vectorSize;
    }
}
