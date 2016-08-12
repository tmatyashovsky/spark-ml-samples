package com.lohika.morning.ml.spark.driver.service.lyrics;

import java.nio.file.Paths;
import java.util.Arrays;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.*;
import org.apache.spark.mllib.feature.Stemmer;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Window;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class TextService {

    @Autowired
    private SparkSession sparkSession;

    private Dataset<Row> getPopMusic(String inputDirectory) {
        Dataset<String> madonnaLyrics = getLyrics(inputDirectory, "madonna.txt");
        Dataset<String> jenniferLopezLyrics = getLyrics(inputDirectory, "jennifer_lopez.txt");
        Dataset<String> britneySpearsLyrics = getLyrics(inputDirectory, "britney_spears.txt");
        Dataset<String> backstreetBoysLyrics = getLyrics(inputDirectory, "backstreet_boys.txt");
        Dataset<String> christinaAguileraLyrics = getLyrics(inputDirectory, "christina_aguilera.txt");
        Dataset<String> modernTalkingLyrics = getLyrics(inputDirectory, "modern_talking.txt");
        Dataset<String> abbaLyrics = getLyrics(inputDirectory, "abba.txt");
        Dataset<String> aceOfBaseLyrics = getLyrics(inputDirectory, "ace_of_base.txt");

        Dataset<String> popLyrics = madonnaLyrics
                                        .union(jenniferLopezLyrics)
                                        .union(britneySpearsLyrics)
                                        .union(backstreetBoysLyrics)
                                        .union(christinaAguileraLyrics)
                                        .union(modernTalkingLyrics)
                                        .union(abbaLyrics)
                                        .union(aceOfBaseLyrics);

        Dataset<Row> popMusic = popLyrics.withColumn("label", functions.lit(1D));
        System.out.println("Pop music sentences = " + popMusic.count());

        return popMusic;
    }

    private Dataset<Row> getMetalMusic(String inputDirectory) {
        Dataset<String> blackSabbathLyrics = getLyrics(inputDirectory, "black_sabbath.txt");
        Dataset<String> metallicaLyrics = getLyrics(inputDirectory, "metallica.txt");
        Dataset<String> moonspellLyrics = getLyrics(inputDirectory, "moonspell.txt");
        Dataset<String> ironMaidenLyrics = getLyrics(inputDirectory, "iron_maiden.txt");
        Dataset<String> inFlamesLyrics = getLyrics(inputDirectory, "in_flames.txt");
        Dataset<String> sentencedLyrics = getLyrics(inputDirectory, "sentenced.txt");
        Dataset<String> nightwishLyrics = getLyrics(inputDirectory, "nightwish.txt");
        Dataset<String> sepulturaLyrics = getLyrics(inputDirectory, "sepultura.txt");

        Dataset<String> metalLyrics = blackSabbathLyrics
                                                .union(metallicaLyrics)
                                                .union(moonspellLyrics)
                                                .union(ironMaidenLyrics)
                                                .union(inFlamesLyrics)
                                                .union(sentencedLyrics)
                                                .union(nightwishLyrics)
                                                .union(sepulturaLyrics);

        Dataset<Row> metalMusic = metalLyrics.withColumn("label", functions.lit(0D));
        System.out.println("Metal music sentences = " + metalLyrics.count());

        return metalMusic;
    }

    public void classifyLyricsWithPipeline(final String inputDirectory) {
        Dataset<Row> sentences = getPopMusic(inputDirectory).union(getMetalMusic(inputDirectory));

        // Remove all punctuation symbols.
        Cleanser cleanser = new Cleanser();

        // Add id and rowNumber based on it.
        Numerator numerator = new Numerator();

        // Split into words.
        Tokenizer tokenizer = new Tokenizer().setInputCol("clean").setOutputCol("words");

        // Remove stop words.
        StopWordsRemover stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filteredWords");

        // Create as many rows as words. This is needed or Stemmer.
        Exploder exploder = new Exploder();

        // Perform stemming.
        Stemmer stemmer = new Stemmer().setInputCol("filteredWord").setOutputCol("stemmedWord").setLanguage("English");

        Uniter uniter = new Uniter();
        Verser verser = new Verser();

        // Create model.
        Word2Vec word2Vec = new Word2Vec().setInputCol("verses").setOutputCol("features");

        LogisticRegression logisticRegression = new LogisticRegression();

        Pipeline pipeline = new Pipeline().setStages(
                                                new PipelineStage[]{
                                                        cleanser,
                                                        numerator,
                                                        tokenizer,
                                                        stopWordsRemover,
                                                        exploder,
                                                        stemmer,
                                                        uniter,
                                                        verser,
                                                        word2Vec,
                                                        logisticRegression});

        // Use a ParamGridBuilder to construct a grid of parameters to search over.
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(new IntParam("sentencesInVerse", "sentencesInVerse", ""), new int[]{2, 4, 8, 16})
                .addGrid(word2Vec.vectorSize(), new int[] {20, 50, 100, 150, 200, 250, 300})
                .addGrid(logisticRegression.regParam(), new double[] {1D, 0.1D, 0.05D})
                .addGrid(logisticRegression.maxIter(), new int[] {100, 200, 300})
                .build();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new BinaryClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid).setNumFolds(3);

        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel model = crossValidator.fit(sentences);

        Arrays.sort(model.avgMetrics());
        System.out.println("Best avg metrics = " + model.avgMetrics()[model.avgMetrics().length - 1]);

        Transformer[] stages = ((PipelineModel)(model.bestModel())).stages();
        System.out.println("Best sentences in verse = " + ((Verser) stages[7]).getSentencesInVerse());
        System.out.println("Best vector size = "  + ((Word2VecModel) stages[8]).getVectorSize());
        System.out.println("Best reg parameter = " + ((LogisticRegressionModel) stages[9]).getRegParam());
        System.out.println("Best max iterations = " + ((LogisticRegressionModel) stages[9]).getMaxIter());
    }

//    private Dataset<Row> getValidationSet(Word2VecModel word2VecModel, int vectorSize, boolean includeVariances) {
//        List<Row> unknownLyrics = Arrays.asList(
//                RowFactory.create("The day that never comes"),
//                RowFactory.create("Is this the end of the beginning"),
//                RowFactory.create("Lost in time I wonder will my ship be found")
//        );
//
//        StructType schema = new StructType(new StructField[]{
//                new StructField("value", DataTypes.StringType, true, Metadata.empty())
//        });
//
//        Dataset<Row> unknownLyricsDataset = sparkSession.createDataFrame(unknownLyrics, schema);
//        Dataset<Row> separatedWords = getWord2VecDataset(unknownLyricsDataset);
//        Dataset<Row> wordsAsFeatures = word2VecModel.transform(separatedWords);
//
//        return getFeatures(wordsAsFeatures, vectorSize, includeVariances);
//    }

    private Dataset<String> getLyrics(String inputDirectory, String fileName) {
        Dataset<String> lyrics = sparkSession.read().textFile(Paths.get(inputDirectory).resolve(fileName).toString());
        lyrics = lyrics.filter(lyrics.col("value").notEqual(""));
        lyrics = lyrics.filter(lyrics.col("value").contains(" "));

        return lyrics;
    }

    public void classifyLyricsWithoutPipeline(final String inputDirectory) {
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
        stemmedSentences.count();

        Column verseSplitExpression = functions
                .when(
                        functions
                                .column("rowNumber")
                                // TODO: using pipeline instead in order to test different values.
                                .mod(4)
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

        // Create model.
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("verses")
                .setOutputCol("features")
                // TODO: using pipeline instead in order to test different values.
                .setVectorSize(100)
                .setMaxSentenceLength(10)
                .setMinCount(0);

        // Fit model.
        Word2VecModel word2VecModel = word2Vec.fit(verses);
        System.out.println("Word2Vec vocabulary = " + word2VecModel.getVectors().count());

        // Train verses and get verses as features.
        Dataset<Row> versesAsFeatures = word2VecModel.transform(verses);

        // Create logistic regression and evaluate it separately to the whole pipeline.
        LogisticRegression logisticRegression = new LogisticRegression();

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(logisticRegression.regParam(), new double[] {1D, 0.1D, 0.05D})
                .addGrid(logisticRegression.maxIter(), new int[] {100, 200, 300})
                .build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(logisticRegression)
                .setEvaluator(new BinaryClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setTrainRatio(0.7);

        TrainValidationSplitModel model = trainValidationSplit.fit(versesAsFeatures);
        Arrays.sort(model.validationMetrics());
        System.out.println("Best validation metrics = " + model.validationMetrics()[model.validationMetrics().length - 1]);
        LogisticRegressionModel logisticRegressionModel = (LogisticRegressionModel) model.bestModel();

        System.out.println("Best reg parameter = " + logisticRegressionModel.getRegParam());
        System.out.println("Best max iterations = " + logisticRegressionModel.getMaxIter());
    }

}
