package com.lohika.morning.ml.spark.driver.service.lyrics;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.MLService;
import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;
import java.nio.file.Paths;
import java.util.*;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.*;
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

    public Dataset<Row> readLyricsFromDirectory(String lyricsInputDirectory) {
        Dataset input = readLyricsForGenre(lyricsInputDirectory, Genre.METAL)
                                                .union(readLyricsForGenre(lyricsInputDirectory, Genre.POP));
        // Reduce the input amount of partition minimal amount (spark.default.parallelism OR 2, whatever is less)
        input = input.coalesce(sparkSession.sparkContext().defaultMinPartitions()).cache();
        // Force caching.
        input.count();

        return input;
    }

    private Dataset<Row> readLyricsForGenre(String inputDirectory, Genre genre) {
        Dataset<Row> lyrics = readLyricsFromDirectory(inputDirectory, genre.name().toLowerCase() + "/*");
        Dataset<Row> labeledLyrics = lyrics.withColumn(LABEL.getName(), functions.lit(genre.getValue()));

        System.out.println(genre.name() + " music sentences = " + lyrics.count());

        return labeledLyrics;
    }

    private Dataset<Row> readLyricsFromDirectory(String inputDirectory, String path) {
        Dataset<String> rawLyrics = sparkSession.read().textFile(Paths.get(inputDirectory).resolve(path).toString());
        rawLyrics = rawLyrics.filter(rawLyrics.col(VALUE.getName()).notEqual(""));
        rawLyrics = rawLyrics.filter(rawLyrics.col(VALUE.getName()).contains(" "));

        // Add source filename column as a unique id.
        Dataset<Row> lyrics = rawLyrics.withColumn(ID.getName(), functions.input_file_name());

        return lyrics;
    }

    public Map<String, Object> classifyLyricsUsingLogisticRegression(final String lyricsInputDirectory, final String modelOutputDirectory) {
        Dataset sentences = readLyricsFromDirectory(lyricsInputDirectory);

        // Remove all punctuation symbols.
        Cleanser cleanser = new Cleanser();

        // Add rowNumber based on it.
        Numerator numerator = new Numerator();

        // Split into words.
        Tokenizer tokenizer = new Tokenizer()
                                    .setInputCol(CLEAN.getName())
                                    .setOutputCol(WORDS.getName());

        // Remove stop words.
        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol(WORDS.getName())
                .setOutputCol(FILTERED_WORDS.getName());

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
                .addGrid(verser.sentencesInVerse(), new int[]{16})
                .addGrid(word2Vec.vectorSize(), new int[] {300})
                .addGrid(logisticRegression.regParam(), new double[] {0.01D})
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

        return getLogisticRegresssionPipelineStatistics(model);
    }

    public Map<String, Object> classifyLyricsUsingNaiveBayes(final String lyricsInputDirectory, final String modelOutputDirectory) {
        Dataset sentences = readLyricsFromDirectory(lyricsInputDirectory);

        // Remove all punctuation symbols.
        Cleanser cleanser = new Cleanser();

        // Add rowNumber based on it.
        Numerator numerator = new Numerator();

        // Split into words.
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol(CLEAN.getName())
                .setOutputCol(WORDS.getName());

        // Remove stop words.
        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol(WORDS.getName())
                .setOutputCol(FILTERED_WORDS.getName());

        // Create as many rows as words. This is needed or Stemmer.
        Exploder exploder = new Exploder();

        // Perform stemming.
        Stemmer stemmer = new Stemmer();

        Uniter uniter = new Uniter();
        Verser verser = new Verser();

        CountVectorizer countVectorizer = new CountVectorizer().setInputCol(VERSES.getName()).setOutputCol("features");

        NaiveBayes naiveBayes = new NaiveBayes();

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
                        countVectorizer,
                        naiveBayes});

        // Use a ParamGridBuilder to construct a grid of parameters to search over.
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{4, 8, 16})
                .build();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new BinaryClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(10);

        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel model = crossValidator.fit(sentences);

        mlService.saveModel(model, modelOutputDirectory);

        return getNaiveBayesPipelineStatistics(model);
    }

    public GenrePrediction predict(final String unknownLyrics, final String modelDirectory) {
        List<Row> unknownLyricsList = Collections.singletonList(
            RowFactory.create(unknownLyrics, Genre.UNKNOWN.getValue())
        );

        StructType schema = new StructType(new StructField[]{
            VALUE.getStructType(),
            LABEL.getStructType()
        });

        Dataset<Row> unknownLyricsDataset = sparkSession.createDataFrame(unknownLyricsList, schema);

        CrossValidatorModel model = mlService.loadCrossValidationModel(modelDirectory);
        getLogisticRegresssionPipelineStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();

        Dataset<Row> predictionsDataset = bestModel.transform(unknownLyricsDataset);
        Row predictionRow = predictionsDataset.first();

        final DenseVector probability = predictionRow.getAs("probability");
        final Double prediction = predictionRow.getAs("prediction");

        System.out.println("\n------------------------------------------------");
        System.out.println("Probability: " + probability);
        System.out.println("Prediction: " + Double.toString(prediction));
        System.out.println("------------------------------------------------\n");

        return new GenrePrediction(getGenre(prediction).getName(), probability.apply(0), probability.apply(1));
    }

    private Map<String, Object> getLogisticRegresssionPipelineStatistics(CrossValidatorModel model) {
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

    private Map<String, Object> getNaiveBayesPipelineStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();

        Arrays.sort(model.avgMetrics());
        modelStatistics.put("Best avg metrics", model.avgMetrics()[model.avgMetrics().length - 1]);

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        modelStatistics.put("Best sentences in verse", ((Verser) stages[7]).getSentencesInVerse());
        modelStatistics.put("Vocabulary", ((CountVectorizerModel) stages[8]).getVocabSize());

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    private void printModelStatistics(Map<String, Object> modelStatistics) {
        System.out.println("\n------------------------------------------------");
        System.out.println("Model statistics:");
        System.out.println(modelStatistics);
        System.out.println("------------------------------------------------\n");
    }

    private Genre getGenre(Double value) {
        for (Genre genre: Genre.values()){
            if (genre.getValue().equals(value)) {
                return genre;
            }
        }

        return Genre.UNKNOWN;
    }

}
