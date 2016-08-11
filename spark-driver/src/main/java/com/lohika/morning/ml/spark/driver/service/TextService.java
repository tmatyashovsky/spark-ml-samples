package com.lohika.morning.ml.spark.driver.service;

import com.lohika.morning.ml.spark.distributed.library.function.map.generic.*;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.mllib.feature.Stemmer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Aggregator;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import scala.Tuple2;

@Component
public class TextService {

    @Autowired
    private SparkSession sparkSession;

    @Autowired
    private MLlibService mLlibService;

    public void classifyDarkLyrics(final String inputDirectory, final int vectorSize, final int sentencesInVerse, final boolean includeVariances) {
        JavaRDD<LabeledPoint> trainingSet = getTrainingSet(inputDirectory, vectorSize, sentencesInVerse, includeVariances);

        Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> datasets = mLlibService.getTrainingAndTestDatasets(trainingSet);

        mLlibService.trainLogisticRegression(datasets._1(), datasets._2(), 2);

//        System.out.print(logisticRegressionModel.predict(mlVectorToMLlibVector(getValidationSet())).collect());
    }

    private Dataset<Row> getPopMusic(String inputDirectory) {
        Dataset<String> madonnaLyrics = getLyrics(inputDirectory, "madonna.txt");
        Dataset<String> jenniferLopezLyrics = getLyrics(inputDirectory, "jennifer_lopez.txt");
        Dataset<String> britneySpearsLyrics = getLyrics(inputDirectory, "britney_spears.txt");
        Dataset<String> backstreetBoysLyrics = getLyrics(inputDirectory, "backstreet_boys.txt");
        Dataset<String> christinaAguileraLyrics = getLyrics(inputDirectory, "christina_aguilera.txt");

        Dataset<String> popLyrics = madonnaLyrics
                                        .union(jenniferLopezLyrics)
                                        .union(britneySpearsLyrics)
                                        .union(backstreetBoysLyrics)
                                        .union(christinaAguileraLyrics);

        Dataset<Row> popMusic = popLyrics.withColumn("label", functions.lit(1D));
        System.out.println("Pop music sentences = " + popMusic.count());

        return popMusic;
    }

    private Dataset<Row> getMetalMusic(String inputDirectory) {
        Dataset<String> blackSabbathLyrics = getLyrics(inputDirectory, "black_sabbath.txt");
        Dataset<String> metallicaLyrics = getLyrics(inputDirectory, "metallica.txt");
        Dataset<String> moonspellLyrics = getLyrics(inputDirectory, "moonspell.txt");
        Dataset<String> ironMaidenLyrics = getLyrics(inputDirectory, "iron_maiden.txt");

        Dataset<String> metalLyrics = blackSabbathLyrics
                                                .union(metallicaLyrics)
                                                .union(moonspellLyrics)
                                                .union(ironMaidenLyrics);

        Dataset<Row> metalMusic = metalLyrics.withColumn("label", functions.lit(0D));
        System.out.println("Metal music sentences = " + metalLyrics.count());

        return metalMusic;
    }

    private JavaRDD<LabeledPoint> getTrainingSet(String inputDirectory, int vectorSize, int sentencesInVerse, boolean includeVariances) {
        Dataset<Row> sentences = getPopMusic(inputDirectory).union(getMetalMusic(inputDirectory));

        // Remove all punctuation symbols.
        sentences = sentences.withColumn("value", functions.regexp_replace(sentences.col("value"), "[^\\w\\s]", ""));

        // Add unique id to each sentence of lyrics.
        Dataset<Row> sentencesWithIds = sentences.withColumn("id", functions.monotonically_increasing_id());
        sentencesWithIds = sentencesWithIds.withColumn("rowNumber", functions.row_number().over(Window.orderBy("id")));

        // Split into words.
        Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words");
        Dataset<Row> words = tokenizer.transform(sentencesWithIds);

        // Remove stop words.
        StopWordsRemover stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filteredWords");
        Dataset<Row> filtered = stopWordsRemover.transform(words);

        // Create as many rows as words. This is needed or Stemmer.
        Column filteredArray = functions.explode(filtered.col("filteredWords"));
        Dataset<Row> filteredWords = filtered.withColumn("filteredWord", filteredArray);

        // Perform stemming.
        Dataset<Row> stemmedWords = new Stemmer()
                .setInputCol("filteredWord")
                .setOutputCol("stemmedWord")
                .setLanguage("English")
                .transform(filteredWords);

        // Unite stemmed words into a sentence again.
        Dataset<Row> stemmedSentences = stemmedWords.groupBy("rowNumber", "value", "label").agg(functions.column("rowNumber"), functions.concat_ws(" ", functions.collect_list("stemmedWord")).as("stemmedSentence"));

        stemmedSentences.cache();

        // Wrap string into array. This is a requirement for Word2Vec input.
        Dataset<Row> word2VecDataset = stemmedSentences.withColumn("sentence",  functions.split(stemmedSentences.col("stemmedSentence"), " "));

        // Create model.
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("sentence")
                .setOutputCol("features")
                .setVectorSize(vectorSize)
                .setMaxSentenceLength(10)
                .setMinCount(0);

        // Fit model.
        Word2VecModel word2VecModel = word2Vec.fit(word2VecDataset);

        Column verseSplitExpression = functions
                .when(
                        functions
                                .column("rowNumber")
                                .mod(sentencesInVerse)
                                .equalTo(1),
                        1
                )
                .otherwise(0);

        Dataset<Row> sententencesPreparedForSplit = stemmedSentences.withColumn("verseStart", verseSplitExpression);

        Dataset<Row> verses = sententencesPreparedForSplit
                .withColumn("verseId",
                        functions
                                .sum("verseStart")
                                .over(
                                        Window
                                                .orderBy("rowNumber")
                                                .rowsBetween(Long.MIN_VALUE, 0)
                                )
                )
                .select("rowNumber", "verseId", "label", "stemmedSentence");

        verses = verses.groupBy("verseId").agg(
                functions.first("rowNumber").as("rowIdForVerse"),
                functions.first("label").as("label"),
                functions.split(functions.concat_ws(" ", functions.collect_list(functions.column("stemmedSentence"))), " ").as("verses")
        );

        word2VecModel.setInputCol("verses");
        // Train words and get words as features features.
        Dataset<Row> versesAsFeatures = word2VecModel.transform(verses);

        return mlLabeledPointToMLlibLabeledPoint(versesAsFeatures, includeVariances);
    }

    private Dataset<Row> getWord2VecDataset(Dataset<Row> sentences) {
        // Add unique id to each sentence of lyrics.
        Dataset<Row> sentencesWithIds = sentences.withColumn("id", functions.monotonically_increasing_id());
        sentencesWithIds = sentences.withColumn("sentence", functions.split(sentencesWithIds.col("value"), " "));

        // Split into words.
        Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words");
        Dataset<Row> words = tokenizer.transform(sentencesWithIds);

        // Create as many row as words. Using this for Word2Vec.
        Dataset<Row> separatedWords = words.withColumn("words", functions.explode(words.col("words")));
        // Wrap string into array. This is a requirement for Word2Vec.
        separatedWords = separatedWords.withColumn("words", functions.array(separatedWords.col("words")));

        return separatedWords;
    }

    private Dataset<Row> getValidationSet(Word2VecModel word2VecModel, int vectorSize, boolean includeVariances) {
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
        Dataset<Row> wordsAsFeatures = word2VecModel.transform(separatedWords);

        return getFeatures(wordsAsFeatures, vectorSize, includeVariances);
    }

    private Dataset<String> getLyrics(String inputDirectory, String fileName) {
        Dataset<String> lyrics = sparkSession.read().textFile(Paths.get(inputDirectory).resolve(fileName).toString());
        lyrics = lyrics.filter(lyrics.col("value").notEqual(""));
        lyrics = lyrics.filter(lyrics.col("value").contains(" "));

        return lyrics;
    }

    public Dataset<Row> getFeatures(Dataset<Row> wordsAsFeatures, int vectorSize, boolean includeVariances) {
//        wordsAsFeatures = wordsAsFeatures.map(new AverageMapFunction(),
//                            RowEncoder.apply(
//                                    wordsAsFeatures.schema()
//                                            .add(
//                                                DataTypes.createStructField("averagePerWord", DataTypes.DoubleType, false))
//                                            .add(
//                                                DataTypes.createStructField("delta", new VectorUDT(), false))
//                                            .add(
//                                                DataTypes.createStructField("cov", DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.DoubleType)), false))));
        Dataset<Row> averagePerSentence = getAverages(wordsAsFeatures, vectorSize);

        if (includeVariances) {
            Dataset<Row> joined = wordsAsFeatures.join(averagePerSentence, "id");
            return getVariances(joined, vectorSize);
        }

        return averagePerSentence;
    }

    public Dataset<Row> getAverages(Dataset<Row> dataset, int vectorSize) {
        Aggregator<Row, DoubleArrayAVGHolder, Row> averageAggregator = new DenseVectorValuesElementsAverageAggregator(vectorSize, true);

        return dataset.groupBy("id").agg(averageAggregator.toColumn().as("averagePerSentence"))
                .select("id", "averagePerSentence.label", "averagePerSentence.averages");
    }

    public Dataset<Row> getVariances(Dataset<Row> dataset, int vectorSize) {
        Aggregator<Row, DoubleArrayVarianceHolder, Row> varianceAggregator = new DenseVectorValuesElementsVarianceAggregator(vectorSize);

        return dataset.groupBy("id").agg(varianceAggregator.toColumn().as("fullFeatures"))
                .select("id", "fullFeatures.label", "fullFeatures.averages", "fullFeatures.variances");
    }

    public JavaRDD<LabeledPoint> mlLabeledPointToMLlibLabeledPoint(Dataset<Row> mlLabeledPointDataset, boolean includeVariances) {
        return mlLabeledPointDataset.javaRDD().map(new ConvertMLLabeledPointToMLlibLabeledPoint(includeVariances));
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
