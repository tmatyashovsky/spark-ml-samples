package com.lohika.morning.ml.spark.driver.service.lyrics;

import com.lohika.morning.ml.spark.driver.service.MLService;
import java.nio.file.Paths;
import java.util.*;
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
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class TextService {

    @Autowired
    private SparkSession sparkSession;

    @Autowired
    private MLService mlService;

    private Dataset<Row> getPopMusic(String inputDirectory) {
        Dataset<String> madonnaLyrics = getLyrics(inputDirectory, "madonna.txt");
        Dataset<String> jenniferLopezLyrics = getLyrics(inputDirectory, "jennifer_lopez.txt");
        Dataset<String> britneySpearsLyrics = getLyrics(inputDirectory, "britney_spears.txt");
        Dataset<String> backstreetBoysLyrics = getLyrics(inputDirectory, "backstreet_boys.txt");
        Dataset<String> christinaAguileraLyrics = getLyrics(inputDirectory, "christina_aguilera.txt");
        Dataset<String> modernTalkingLyrics = getLyrics(inputDirectory, "modern_talking.txt");
        Dataset<String> abbaLyrics = getLyrics(inputDirectory, "abba.txt");
        Dataset<String> aceOfBaseLyrics = getLyrics(inputDirectory, "ace_of_base.txt");
        Dataset<String> eTypeLyrics = getLyrics(inputDirectory, "e-type.txt");
        Dataset<String> michaelJacksonLyrics = getLyrics(inputDirectory, "michael_jackson.txt");
        Dataset<String> mariahCareyLyrics = getLyrics(inputDirectory, "mariah_carey.txt");
        Dataset<String> spiceGirlsLyrics = getLyrics(inputDirectory, "spice_girls.txt");

        Dataset<String> popLyrics = madonnaLyrics
                                        .union(jenniferLopezLyrics)
                                        .union(britneySpearsLyrics)
                                        .union(backstreetBoysLyrics)
                                        .union(christinaAguileraLyrics)
                                        .union(modernTalkingLyrics)
                                        .union(abbaLyrics)
                                        .union(aceOfBaseLyrics)
                                        .union(mariahCareyLyrics)
                                        .union(spiceGirlsLyrics)
                                        .union(eTypeLyrics)
                                        .union(michaelJacksonLyrics);

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
        Dataset<String> marilynMansonLyrics = getLyrics(inputDirectory, "marilyn_manson.txt");
        Dataset<String> megadethLyrics = getLyrics(inputDirectory, "megadeth.txt");
        Dataset<String> darkTranquillityLyrics = getLyrics(inputDirectory, "dark_tranquillity.txt");
        Dataset<String> helloweenLyrics = getLyrics(inputDirectory, "helloween.txt");
        Dataset<String> ozzyOzbourneLyrics = getLyrics(inputDirectory, "ozzy_ozbourne.txt");

        Dataset<String> metalLyrics = blackSabbathLyrics
                                                .union(metallicaLyrics)
                                                .union(moonspellLyrics)
                                                .union(ironMaidenLyrics)
                                                .union(inFlamesLyrics)
                                                .union(sentencedLyrics)
                                                .union(nightwishLyrics)
                                                .union(sepulturaLyrics)
                                                .union(marilynMansonLyrics)
                                                .union(megadethLyrics)
                                                .union(darkTranquillityLyrics)
                                                .union(helloweenLyrics)
                                                .union(ozzyOzbourneLyrics);

        Dataset<Row> metalMusic = metalLyrics.withColumn("label", functions.lit(0D));
        System.out.println("Metal music sentences = " + metalLyrics.count());

        return metalMusic;
    }

    public Map<String, Object> classifyLyricsWithPipeline(final String lyricsInputDirectory, final String modelOutputDirectory) {
        Dataset<Row> sentences = getPopMusic(lyricsInputDirectory).union(getMetalMusic(lyricsInputDirectory));

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
        Stemmer stemmer = new Stemmer();

        Uniter uniter = new Uniter();
        Verser verser = new Verser();

        // Create model.
        Word2Vec word2Vec = new Word2Vec().setInputCol("verses").setOutputCol("features").setMinCount(0);

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
                .addGrid(verser.sentencesInVerse(), new int[]{2, 4, 8, 16, 32})
                .addGrid(word2Vec.vectorSize(), new int[] {50, 100, 150, 200, 250, 300})
                .addGrid(logisticRegression.regParam(), new double[] {0.05D})
                .addGrid(logisticRegression.maxIter(), new int[] {200})
                .build();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new BinaryClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(10);

        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel model = crossValidator.fit(sentences);

        mlService.saveModel(model, modelOutputDirectory);

        return getModelStatistics(model);
    }

    public double predict(final String unknownLyrics, final String modelDirectory) {
        List<Row> unknownLyricsList = Collections.singletonList(
            RowFactory.create(unknownLyrics, -1.0D)
        );

        StructType schema = new StructType(new StructField[]{
            DataTypes.createStructField("value", DataTypes.StringType, false),
            DataTypes.createStructField("label", DataTypes.DoubleType, false)
        });

        Dataset<Row> unknownLyricsDataset = sparkSession.createDataFrame(unknownLyricsList, schema);

        CrossValidatorModel model = mlService.loadCrossValidationModel(modelDirectory);
        getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();

        Dataset<Row> predictions = bestModel.transform(unknownLyricsDataset);
        Row prediction = predictions.first();

        System.out.println("\n------------------------------------------------");
        System.out.println("Probability: " + prediction.getAs("probability"));
        System.out.println("Prediction: " + Double.toString(prediction.getAs("prediction")));
        System.out.println("------------------------------------------------\n");

        return prediction.getAs("prediction");
    }

    private Dataset<String> getLyrics(String inputDirectory, String fileName) {
        Dataset<String> lyrics = sparkSession.read().textFile(Paths.get(inputDirectory).resolve(fileName).toString());
        lyrics = lyrics.filter(lyrics.col("value").notEqual(""));
        lyrics = lyrics.filter(lyrics.col("value").contains(" "));

        return lyrics;
    }

    public TrainValidationSplitModel classifyLyricsWithoutPipeline(final String inputDirectory) {
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
        Dataset<Row> stemmedWords = new org.apache.spark.mllib.feature.Stemmer()
                .setInputCol("filteredWord")
                .setOutputCol("stemmedWord")
                .setLanguage("English")
                .transform(filteredWords);

        // Unite stemmed words into a sentence again.
        Dataset<Row> stemmedSentences = stemmedWords.groupBy("rowNumber", "value", "label").agg(functions.column("rowNumber"), functions.concat_ws(" ", functions.collect_list("stemmedWord")).as("stemmedSentence"));

        stemmedSentences.cache();
        stemmedSentences.count();

        // Wrap string into array. This is a requirement for Word2Vec input.
        Dataset<Row> word2VecDataset = stemmedSentences.withColumn("sentence",  functions.split(stemmedSentences.col("stemmedSentence"), " "));

        // Create model.
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("sentence")
                .setOutputCol("features")
                // TODO: using pipeline instead in order to test different values.
                .setVectorSize(200)
                .setMaxSentenceLength(10)
                .setMinCount(0);

        // Fit model.
        Word2VecModel word2VecModel = word2Vec.fit(word2VecDataset);
        System.out.println("Word2Vec vocabulary = " + word2VecModel.getVectors().count());

        Column verseSplitExpression = functions
                .when(
                        functions
                                .column("rowNumber")
                                // TODO: using pipeline instead in order to test different values.
                                .mod(16)
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

        // Train verses and get verses as features.
        Dataset<Row> versesAsFeatures = word2VecModel.transform(verses);

        // Create logistic regression and evaluate it separately to the whole pipeline.
        LogisticRegression logisticRegression = new LogisticRegression();

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(logisticRegression.regParam(), new double[] {1D})
                .addGrid(logisticRegression.maxIter(), new int[] {10})
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

        return model;
    }

    private Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();

        Arrays.sort(model.avgMetrics());
        modelStatistics.put("Best avg metrics", model.avgMetrics()[model.avgMetrics().length - 1]);

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        modelStatistics.put("Best sentences in verse", ((Verser) stages[7]).getSentencesInVerse());
        modelStatistics.put("Word2Vec vocabulary", ((Word2VecModel) stages[8]).getVectors().count());
        modelStatistics.put("Best vector size", ((Word2VecModel) stages[8]).getVectorSize());
        modelStatistics.put("Best reg parameter", ((LogisticRegressionModel) stages[9]).getRegParam());
        modelStatistics.put("Best max iterations", ((LogisticRegressionModel) stages[9]).getMaxIter());

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    private void printModelStatistics(Map<String, Object> modelStatistics) {
        System.out.println("\n------------------------------------------------");
        System.out.println("Model statistics:");
        System.out.println(modelStatistics);
        System.out.println("------------------------------------------------\n");
    }

}
