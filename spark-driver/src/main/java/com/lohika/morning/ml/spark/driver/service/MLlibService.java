package com.lohika.morning.ml.spark.driver.service;

import com.lohika.morning.ml.spark.distributed.library.function.verify.VerifyLogisticRegressionModel;
import com.lohika.morning.ml.spark.distributed.library.function.verify.VerifyNaiveBayesModel;
import com.lohika.morning.ml.spark.distributed.library.function.verify.VerifySVMModel;
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
    private MLlibUtilityService utilityService;

    public void trainLogisticRegression(String trainingSetParquetFilePath, String testSetParquetFilePath, int numClasses) {
        Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> trainingAndTestDataSet = getTrainingAndTestDatasets(
            trainingSetParquetFilePath, testSetParquetFilePath);

        trainLogisticRegression(trainingAndTestDataSet._1(), trainingAndTestDataSet._2(), numClasses);
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

        JavaRDD<LabeledPoint> fullSet = utilityService.parquetToLabeledPoint(fullSetDataset);

        // Split initial RDD into two... [70% training data, 30% testing data].
        JavaRDD<LabeledPoint> trainingSet = fullSet.sample(false, 0.7, 0L);
        trainingSet.cache();
        trainingSet.count();

        JavaRDD<LabeledPoint> testSet = fullSet.subtract(trainingSet);

        return new Tuple2<>(trainingSet, testSet);
    }

    private Tuple2<JavaRDD<LabeledPoint>, JavaRDD<LabeledPoint>> getTrainingAndTestDatasets(
            final String trainingSetParquetFilePath, final String testSetParquetFilePath) {
        Dataset<Row> trainingSetDataFrame =sparkSession.read().parquet(trainingSetParquetFilePath);
        trainingSetDataFrame.cache();
        trainingSetDataFrame.count();

        JavaRDD<LabeledPoint> trainingSet = utilityService.parquetToLabeledPoint(trainingSetDataFrame);

        Dataset<Row> testSetDataFrame = sparkSession.read().parquet(testSetParquetFilePath);

        JavaRDD<LabeledPoint> testSet = utilityService.parquetToLabeledPoint(testSetDataFrame);

        return new Tuple2<>(trainingSet, testSet);
    }

    private void trainLogisticRegression(JavaRDD<LabeledPoint> trainingSet, JavaRDD<LabeledPoint> testSet, int numClasses) {
        // Run training algorithm to build the model.
        final LogisticRegressionModel logisticRegression =  new LogisticRegressionWithLBFGS()
                .setNumClasses(numClasses)
                .run(trainingSet.rdd());

        JavaPairRDD<Object, Object> predictionAndLabels = testSet.mapToPair(
            new VerifyLogisticRegressionModel(logisticRegression));

        System.out.println("Logistic regression precision = " + getMulticlassModelPrecision(predictionAndLabels));
    }

    private void trainNaiveBayes(JavaRDD<LabeledPoint> trainingSet, JavaRDD<LabeledPoint> testSet) {
        // Run training algorithm to build the model.
        final NaiveBayesModel naiveBayesModel = NaiveBayes.train(trainingSet.rdd(), 1.0);

        JavaPairRDD<Object, Object> predictionAndLabels = testSet.mapToPair(
            new VerifyNaiveBayesModel(naiveBayesModel));

        System.out.println("Naive Bayes precision = " + getMulticlassModelPrecision(predictionAndLabels));
    }

    private void trainSVM(JavaRDD<LabeledPoint> trainingSet, JavaRDD<LabeledPoint> testSet, int numIterations) {
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

}



