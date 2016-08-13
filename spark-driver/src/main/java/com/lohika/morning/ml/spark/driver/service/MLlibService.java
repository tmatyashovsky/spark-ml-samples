package com.lohika.morning.ml.spark.driver.service;

import com.lohika.morning.ml.spark.distributed.library.function.verify.VerifyLogisticRegressionModel;
import com.lohika.morning.ml.spark.distributed.library.function.verify.VerifyNaiveBayesModel;
import com.lohika.morning.ml.spark.distributed.library.function.verify.VerifySVMModel;
import com.lohika.morning.ml.spark.driver.service.mnist.MnistUtilityService;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.*;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import scala.Tuple2;

@Component
public class MLlibService {

    @Autowired
    private SparkSession sparkSession;

    @Autowired
    private MnistUtilityService mnistUtilityService;

    public Map<String, Object> trainLogisticRegression(String trainingSetParquetFilePath,
                                                       String testSetParquetFilePath,
                                                       int numClasses,
                                                       String modelOutputPath) {
        Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> trainingAndTestDataSet = getTrainingAndTestDatasets(
            trainingSetParquetFilePath, testSetParquetFilePath);

        LogisticRegressionModel logisticRegressionModel = trainLogisticRegression(trainingAndTestDataSet._1(),
                                                                                  trainingAndTestDataSet._2(),
                                                                                  numClasses);

        saveModel(logisticRegressionModel, modelOutputPath);

        return validateLogisticRegression(trainingAndTestDataSet._1(),
                                          trainingAndTestDataSet._2(),
                                          logisticRegressionModel);
    }

    public void trainLogisticRegression(String fullSetParquetFilePath, int numClasses) {
        Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> trainingAndTestDataSet =
            getTrainingAndTestDatasets(fullSetParquetFilePath);

        trainLogisticRegression(trainingAndTestDataSet._1(), trainingAndTestDataSet._2(), numClasses);
    }

    public void trainNaiveBayes(String trainingSetParquetFilePath, String testSetParquetFilePath) {
        Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> trainingAndTestDataSet = getTrainingAndTestDatasets(
                trainingSetParquetFilePath, testSetParquetFilePath);

        trainNaiveBayes(trainingAndTestDataSet._1(), trainingAndTestDataSet._2());
    }

    public void trainNaiveBayes(String fullSetParquetFilePath) {
        Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> trainingAndTestDataSet = getTrainingAndTestDatasets(fullSetParquetFilePath);

        trainNaiveBayes(trainingAndTestDataSet._1(), trainingAndTestDataSet._2());
    }

    public void trainSVM(String trainingSetParquetFilePath, String testSetParquetFilePath, int numIterations) {
        Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> trainingAndTestDataSet = getTrainingAndTestDatasets(
                trainingSetParquetFilePath, testSetParquetFilePath);

        trainSVM(trainingAndTestDataSet._1(), trainingAndTestDataSet._2(), numIterations);
    }

    public void trainSVM(String fullSetParquetFilePath, int numIterations) {
        Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> trainingAndTestDataSet = getTrainingAndTestDatasets(fullSetParquetFilePath);

        trainSVM(trainingAndTestDataSet._1(), trainingAndTestDataSet._2(), numIterations);
    }

    private Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> getTrainingAndTestDatasets(final String fullSetParquetFilePath) {
        Dataset<Row> fullSetDataset = sparkSession.read().parquet(fullSetParquetFilePath);

        JavaRDD<LabeledPoint> fullSet = mnistUtilityService.rowToLabeledPoint(fullSetDataset);

        return getTrainingAndTestDatasets(fullSet);
    }

    public Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> getTrainingAndTestDatasets(JavaRDD<LabeledPoint> fullSet) {
        // Split initial RDD into two... [70% training data, 30% testing data].
        JavaRDD<LabeledPoint> trainingSet = fullSet.sample(false, 0.7, 0L);
        trainingSet.cache();
        trainingSet.count();

        JavaRDD<LabeledPoint> testSet = fullSet.subtract(trainingSet);
        testSet.cache();
        testSet.count();

        return new Tuple2<>(trainingSet, testSet);
    }

    private Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> getTrainingAndTestDatasets(
            final String trainingSetParquetFilePath, final String testSetParquetFilePath) {
        Dataset<Row> trainingSetDataset =sparkSession.read().parquet(trainingSetParquetFilePath);
        trainingSetDataset.cache();
        trainingSetDataset.count();

        JavaRDD<LabeledPoint> trainingSetRDD = mnistUtilityService.rowToLabeledPoint(trainingSetDataset);

        Dataset<Row> testSetDataset = sparkSession.read().parquet(testSetParquetFilePath);

        JavaRDD<LabeledPoint> testSetRDD = mnistUtilityService.rowToLabeledPoint(testSetDataset);

        return new Tuple2<>(trainingSetRDD, testSetRDD);
    }

    public LogisticRegressionModel trainLogisticRegression(JavaRDD<LabeledPoint> trainingSet, JavaRDD<LabeledPoint> testSet, int numClasses) {
        // Run training algorithm to build the model.
        return new LogisticRegressionWithLBFGS()
                .setNumClasses(numClasses)
                .run(trainingSet.rdd());
    }

    private Map<String, Object> validateLogisticRegression(JavaRDD<LabeledPoint> trainingSet, JavaRDD<LabeledPoint> testSet, LogisticRegressionModel logisticRegressionModel) {
        JavaPairRDD<Object, Object> predictionAndLabelsForTestSet = testSet.mapToPair(
            new VerifyLogisticRegressionModel(logisticRegressionModel));

        JavaPairRDD<Object, Object> predictionAndLabelsForTrainingSet = trainingSet.mapToPair(
            new VerifyLogisticRegressionModel(logisticRegressionModel));

        Map<String, Object> modelStatistics = new HashMap<>();
        modelStatistics.put("Logistic regression precision on training set", getMulticlassModelPrecision(predictionAndLabelsForTrainingSet));
        modelStatistics.put("Logistic regression precision on test set", getMulticlassModelPrecision(predictionAndLabelsForTestSet));

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    private void printModelStatistics(Map<String, Object> modelStatistics) {
        System.out.println("\n------------------------------------------------");
        System.out.println("Model statistics:");
        System.out.println(modelStatistics);
        System.out.println("------------------------------------------------\n");
    }

    public void trainNaiveBayes(JavaRDD<LabeledPoint> trainingSet, JavaRDD<LabeledPoint> testSet) {
        // Run training algorithm to build the model.
        final NaiveBayesModel naiveBayesModel = NaiveBayes.train(trainingSet.rdd(), 1.0);

        JavaPairRDD<Object, Object> predictionAndLabels = testSet.mapToPair(
            new VerifyNaiveBayesModel(naiveBayesModel));

        System.out.println("Naive Bayes precision = " + getMulticlassModelPrecision(predictionAndLabels));
    }

    public void trainSVM(JavaRDD<LabeledPoint> trainingSet, JavaRDD<LabeledPoint> testSet, int numIterations) {
        // Run training algorithm to build the model.
        final SVMModel svmModel = SVMWithSGD.train(trainingSet.rdd(), numIterations);

        JavaPairRDD<Object, Object> predictionAndLabels = testSet.mapToPair(
                new VerifySVMModel(svmModel));

        System.out.println("SVM area Under ROC = " + getBinaryClassificationModelPrecision(predictionAndLabels));
    }

    private double getMulticlassModelPrecision(JavaPairRDD<Object, Object> predictionAndLabels) {
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

        return metrics.precision();
    }

    private double getBinaryClassificationModelPrecision(JavaPairRDD<Object, Object> predictionAndLabels) {
        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionAndLabels.rdd());

        return metrics.areaUnderROC();
    }

    public void saveModel(LogisticRegressionModel model, String modelOutputDirectory) {
        model.save(sparkSession.sparkContext(), modelOutputDirectory);

        System.out.println("\n------------------------------------------------");
        System.out.println("Saved logistic regression model to " + modelOutputDirectory);
        System.out.println("------------------------------------------------\n");
    }

    public LogisticRegressionModel loadLogisticRegression(String modelDirectoryPath) {
        LogisticRegressionModel model = LogisticRegressionModel.load(sparkSession.sparkContext(), modelDirectoryPath);

        System.out.println("\n------------------------------------------------");
        System.out.println("Loaded logistic regression model from " + modelDirectoryPath);
        System.out.println("------------------------------------------------\n");
        return model;
    }
}



