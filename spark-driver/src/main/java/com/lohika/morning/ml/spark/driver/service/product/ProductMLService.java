package com.lohika.morning.ml.spark.driver.service.product;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class ProductMLService {

    @Autowired
    private SparkSession sparkSession;

    public Map<String, Object> trainRandomForestClassifier(String trainingSetCsvFilePath, String testSetCsvFilePath) {
        Dataset<Row> trainingSet = readTrainingSet(trainingSetCsvFilePath);

        FeatureExtractor featureExtractor = new FeatureExtractor();
        Dataset<Row> data = featureExtractor.transform(trainingSet);

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(data);

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(10)
                .fit(data);

        RandomForestClassifier randomForestClassifier = new RandomForestClassifier();
        randomForestClassifier.setLabelCol("indexedLabel");
        randomForestClassifier.setFeaturesCol("indexedFeatures");
        randomForestClassifier.setMaxMemoryInMB(512);

        // Convert indexed labels back to original labels.
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels());

        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[] {
                        featureExtractor, labelIndexer, featureIndexer, randomForestClassifier, labelConverter});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(randomForestClassifier.numTrees(), new int[] {20})
                .addGrid(randomForestClassifier.maxDepth(), new int[] {30})
                .build();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel");

        // CrossValidator will try all combinations of values and determine best model using the evaluator.
        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(10);

        CrossValidatorModel model = crossValidator.fit(trainingSet);

        return getPipelineStatistics(model);
    }

    private Dataset<Row> readTrainingSet(String trainingSetCsvFilePath) {
        Dataset<Row> rawTrainingSet = sparkSession
                .read()
                .option("header", "true")
                .schema(getTrainingSetSchema())
                .csv(trainingSetCsvFilePath);

        rawTrainingSet.count();
        rawTrainingSet.cache();

        return rawTrainingSet;
    }

    private Dataset<Row> readTestSet(String testSetCsvFilePath) {
        Dataset<Row> rawTrainingSet = sparkSession
                .read()
                .option("header", "true")
                .schema(getTestSetSchema())
                .csv(testSetCsvFilePath);

        rawTrainingSet.count();
        rawTrainingSet.cache();

        return rawTrainingSet;
    }

    private StructType getTrainingSetSchema() {
        List<StructField> structFields = new ArrayList<>();
        structFields.add(new StructField("id", DataTypes.IntegerType, true, Metadata.empty()));

        IntStream.rangeClosed(1, 93)
                .mapToObj(i -> new StructField("feat_" + i, DataTypes.IntegerType, true, Metadata.empty()))
                .collect(Collectors.toCollection(() -> structFields));

        structFields.add(new StructField("target", DataTypes.StringType, true, Metadata.empty()));

        return new StructType(structFields.toArray(new StructField[95]));
    }

    private StructType getTestSetSchema() {
        List<StructField> structFields = new ArrayList<>();
        structFields.add(new StructField("id", DataTypes.IntegerType, true, Metadata.empty()));

        IntStream.rangeClosed(1, 93)
                .mapToObj(i -> new StructField("feat_" + i, DataTypes.IntegerType, true, Metadata.empty()))
                .collect(Collectors.toCollection(() -> structFields));

        return new StructType(structFields.toArray(new StructField[94]));
    }

    private Map<String, Object> getPipelineStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();

        modelStatistics.put("avgMetrics", model.avgMetrics());
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
