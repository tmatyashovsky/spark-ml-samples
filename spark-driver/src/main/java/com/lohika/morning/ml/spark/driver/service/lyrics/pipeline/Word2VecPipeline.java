package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.lyrics.Genre;
import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;
import com.lohika.morning.ml.spark.driver.service.lyrics.word2vec.Similarity;
import com.lohika.morning.ml.spark.driver.service.lyrics.word2vec.Synonym;
import java.util.*;
import java.util.stream.Collectors;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import scala.collection.mutable.WrappedArray;

@Component
public class Word2VecPipeline extends CommonLyricsPipeline {

    @Autowired
    private SparkSession sparkSession;

    @Value("${lyrics.model.directory.path}")
    private String lyricsModelDirectoryPath;

    public Map<String, Object> train() {
        Dataset<Row> sentences = readLyrics();

        Cleanser cleanser = new Cleanser();

        Numerator numerator = new Numerator();

        Tokenizer tokenizer = new Tokenizer()
                .setInputCol(CLEAN.getName())
                .setOutputCol(WORDS.getName());

        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol(WORDS.getName())
                .setOutputCol(FILTERED_WORDS.getName());

        Exploder exploder = new Exploder();

        Stemmer stemmer = new Stemmer();

        Uniter uniter = new Uniter();
        Verser verser = new Verser();

        // Create model.
        Word2Vec word2Vec = new Word2Vec()
                                    .setInputCol(VERSE.getName())
                                    .setOutputCol("features")
                                    .setVectorSize(300)
                                    .setMinCount(0);

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
                        word2Vec});

        PipelineModel pipelineModel = pipeline.fit(sentences);
        saveModel(pipelineModel, getModelDirectory());

        Map<String, Object> modelStatistics = new HashMap<>();
        Word2VecModel word2VecModel = (Word2VecModel) pipelineModel.stages()[pipelineModel.stages().length - 1];

        modelStatistics.put("Word2Vec vectors count", word2VecModel.getVectors().count());

        return modelStatistics;
    }

    public List<Synonym> findSynonyms(String lyrics) {
        final PipelineModel pipelineModel = PipelineModel.load(getModelDirectory());
        final Word2VecModel word2VecModel = (Word2VecModel) pipelineModel.stages()[8];

        final Dataset<Row> synonyms = word2VecModel.findSynonyms(lyrics, 5);
        return synonyms.collectAsList().stream()
                                       .map(row -> new Synonym(row.getAs("word"), row.getAs("similarity")))
                                       .collect(Collectors.toList());
    }

    public List<Similarity> calculateSimilarity(String lyrics) {
        String verses[] = lyrics.split("\\r?\\n");
        Dataset<String> versesDataset = sparkSession.createDataset(Arrays.asList(verses),
            Encoders.STRING());

        final PipelineModel pipelineModel = PipelineModel.load(getModelDirectory());

        Dataset<Row> dataset = versesDataset
                .withColumn(LABEL.getName(), functions.lit(Genre.UNKNOWN.getValue()))
                .withColumn(ID.getName(), functions.lit("unknown.txt"));

        Dataset<Row> word2Vec = pipelineModel.transform(dataset);
        final List<Row> features = word2Vec.select("verse", "features").collectAsList();

        List<Similarity> similarities = new ArrayList<>();

        for (Row leftFeature : features) {
            for (Row rightFeature : features) {
                similarities.add(new Similarity(getVerse(leftFeature), getVerse(rightFeature),
                                 cosineSimilarity(getVector(leftFeature), getVector(rightFeature))));
            }
        }

        return similarities;
    }

    private double[] getVector(Row row) {
        return ((DenseVector)row.getAs("features")).toArray();
    }

    private String getVerse(Row row) {
        String[] verse = (String[]) ((WrappedArray)row.getAs("verse")).array();

        return Arrays.stream(verse).collect(Collectors.joining(" "));
    }

    private double cosineSimilarity(double[] leftVector, double[] rightVector) {
        final double product = product(leftVector, rightVector);

        double d1 = 0.0d;
        for (final double value : leftVector) {
            d1 += Math.pow(value, 2);
        }

        double d2 = 0.0d;
        for (final double value : rightVector) {
            d2 += Math.pow(value, 2);
        }

        return product / (Math.sqrt(d1) * Math.sqrt(d2));
    }

    private double product(final double[] leftVector, double[] rightVector) {
        double product = 0;

        for (int i=0; i<leftVector.length; i++) {
            product += leftVector[i] * rightVector[i];
        }

        return product;
    }

    public String getModelDirectory() {
        return lyricsModelDirectoryPath + "/word2vec/";
    }

    @Override
    public CrossValidatorModel classify() {
        throw new RuntimeException("Not supported for word2Vec");
    }
}

