package com.lohika.morning.ml.spark.driver.service;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import scala.Tuple2;

@Component
public class MLService {

    @Autowired
    private SparkSession sparkSession;

    public void trainLinearRegression(String fullSetParquetFilePath, int numIterations) {
        Tuple2<Dataset<Row>, Dataset<Row>> trainingAndTestDataSet = getTrainingAndTestDatasets(fullSetParquetFilePath);

        trainLinearRegression(trainingAndTestDataSet._1(), trainingAndTestDataSet._2(), numIterations);
    }

    public void trainLinearRegressionUsingCrossValidator(String fullSetParquetFilePath, int numIterations) {
        trainLinearRegressionUsingCrossValidator(getTrainingDataset(fullSetParquetFilePath), numIterations);
    }

    public void trainKMeans(String fullSetParquetFilePath) {
        Dataset<Row> dataset = getTrainingDataset(fullSetParquetFilePath);

        // Trains a k-means model
        KMeans kmeans = new KMeans().setK(3);
        KMeansModel kMeansModel = kmeans.fit(dataset);

        // Shows the result.
        Vector[] centers = kMeansModel.clusterCenters();
        System.out.println("Cluster Centers: ");
        for (Vector center: centers) {
            System.out.println(center);
        }
    }

    private Tuple2<Dataset<Row>, Dataset<Row>> getTrainingAndTestDatasets(final String fullSetParquetFilePath) {
        Dataset<Row> fullSetDataFrame = sparkSession.read().parquet(fullSetParquetFilePath);

        // Split initial RDD into two... [70% training data, 30% testing data].
        Dataset<Row>[] splitDataFrames = fullSetDataFrame.randomSplit(new double[]{0.7D, 0.3D});
        splitDataFrames[0].cache();
        splitDataFrames[0].count();

        return new Tuple2<>(splitDataFrames[0], splitDataFrames[1]);
    }

    private Dataset<Row> getTrainingDataset(final String fullSetParquetFilePath) {
        Dataset<Row> dataset = sparkSession.read().parquet(fullSetParquetFilePath);

        dataset.cache();
        dataset.count();

        return dataset;
    }

    private void trainLinearRegression(Dataset<Row> trainingSet, Dataset<Row> testSet, int numIterations) {
        LinearRegression linearRegression = new LinearRegression()
                .setMaxIter(numIterations)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        // Fit the model.
        LinearRegressionModel linearRegressionModel = linearRegression.fit(trainingSet);

        // Print the coefficients for linear regression.
        System.out.println("Coefficients: " + linearRegressionModel.coefficients());

        LinearRegressionTrainingSummary trainingSummary = linearRegressionModel.summary();
        System.out.println("Total iterations: " + trainingSummary.totalIterations());
        System.out.println("Root mean squared error: " + trainingSummary.rootMeanSquaredError());

        // Shows predictions for test set.
        Dataset<Row> predictions = linearRegressionModel.transform(testSet);

        Row predictedRow = predictions.select("features", "label", "prediction").first();
        System.out.println("(" + predictedRow.get(0) + ", " + predictedRow.get(1) + ", prediction=" + predictedRow.get(2));
    }

    private void trainLinearRegressionUsingCrossValidator(Dataset<Row> trainingSet, int numIterations) {
        LinearRegression linearRegression = new LinearRegression();

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(linearRegression.regParam(), new double[] {0.3, 0.1})
                .addGrid(linearRegression.fitIntercept())
                .addGrid(linearRegression.elasticNetParam(), new double[] {0.0, 0.5, 1.0})
                .addGrid(linearRegression.maxIter(), new int[] {numIterations})
                .build();

        // CrossValidator will try all combinations of values and determine best model using the evaluator.
        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3);

        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel crossValidatorModel = crossValidator.fit(trainingSet);

        // Print the coefficients for the best linear regression.
        System.out.println("Coefficients: " + ((LinearRegressionModel)crossValidatorModel.bestModel()).coefficients());
    }

}



